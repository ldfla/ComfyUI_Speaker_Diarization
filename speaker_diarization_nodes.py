from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchaudio
from comfy import model_management as mm
from pyannote.audio import Pipeline

@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class SpeakerDiarizerChronoNode:
    """
    Speaker diarization for ComfyUI with chronological output ordering.

    Output mapping:
    - speaker_1_audio = first unique speaker to speak
    - speaker_2_audio = second unique speaker to speak
    - ...
    - speaker_5_audio = fifth unique speaker to speak

    Designed for telephony/call-center scenarios where there may be up to 5
    distinct voices including IVR/URA prompts.
    """

    MAX_OUTPUT_SPEAKERS = 5
    TARGET_SAMPLE_RATE = 16000

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Hugging Face access token for pyannote. "
                            "Required for gated model access."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device used by pyannote pipeline",
                    },
                ),
                "min_speakers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Minimum expected number of speakers",
                    },
                ),
                "max_speakers": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Maximum expected number of speakers",
                    },
                ),
                "merge_gap_ms": (
                    "INT",
                    {
                        "default": 200,
                        "min": 0,
                        "max": 5000,
                        "step": 50,
                        "tooltip": (
                            "Merge adjacent segments of the same speaker when the gap "
                            "between them is less than or equal to this value"
                        ),
                    },
                ),
                "keep_only_detected_speakers": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "If true, only diarized regions are kept in each speaker output. "
                            "If false, output remains isolated by speaker but preserves "
                            "timeline silence everywhere else anyway."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = (
        "speaker_1_audio",
        "speaker_2_audio",
        "speaker_3_audio",
        "speaker_4_audio",
        "speaker_5_audio",
        "summary",
        "segments_json",
    )
    FUNCTION = "diarize_audio"
    CATEGORY = "Audio/Isolation"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[SpeakerDiarizerChronoNode] {message}")

    @staticmethod
    def _round3(value: float) -> float:
        """Round to 3 decimal places without using built-in round (Pyre2 compat)."""
        return int(float(value) * 1000 + 0.5) / 1000

    @staticmethod
    def _normalize_waveform_shape(waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert ComfyUI audio waveform to mono 1D tensor.
        Expected possible shapes:
        - (B, C, S)
        - (C, S)
        - (S,)
        """
        if waveform.ndim == 3:  # (B, C, S)
            return waveform[0].mean(dim=0)
        if waveform.ndim == 2:  # (C, S)
            return waveform.mean(dim=0)
        if waveform.ndim == 1:  # (S,)
            return waveform
        raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")

    @classmethod
    def _make_audio_output(cls, waveform_1d: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor | int]:
        """
        ComfyUI AUDIO format typically expects shape (B, C, S).
        """
        return {
            "waveform": waveform_1d.unsqueeze(0).unsqueeze(0),
            "sample_rate": sample_rate,
        }

    @classmethod
    def _make_silent_output(cls, reference_waveform_1d: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor | int]:
        silent = torch.zeros_like(reference_waveform_1d)
        return cls._make_audio_output(silent, sample_rate)

    @classmethod
    def _silent_outputs(
        cls,
        audio: Dict[str, torch.Tensor | int],
        message: str,
    ) -> Tuple[
        Dict[str, torch.Tensor | int],
        Dict[str, torch.Tensor | int],
        Dict[str, torch.Tensor | int],
        Dict[str, torch.Tensor | int],
        Dict[str, torch.Tensor | int],
        str,
        str,
    ]:
        sample_rate = int(audio["sample_rate"])
        waveform = cls._normalize_waveform_shape(audio["waveform"])  # type: ignore[index]
        silent = cls._make_silent_output(waveform, sample_rate)
        segments_json = json.dumps({"error": message}, ensure_ascii=False, indent=2)
        return (
            silent,
            silent,
            silent,
            silent,
            silent,
            message,
            segments_json,
        )

    @staticmethod
    def _select_device(device: str) -> torch.device:
        if device == "auto":
            return mm.get_torch_device()
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA foi solicitado, mas não está disponível.")
            return torch.device("cuda")
        return torch.device("cpu")

    @classmethod
    def _resample_if_needed(
        cls,
        waveform_1d: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int]:
        if sample_rate == cls.TARGET_SAMPLE_RATE:
            return waveform_1d, sample_rate

        cls._log(f"Resampling {sample_rate}Hz -> {cls.TARGET_SAMPLE_RATE}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=cls.TARGET_SAMPLE_RATE,
        )
        resampled = resampler(waveform_1d)
        return resampled, cls.TARGET_SAMPLE_RATE

    @staticmethod
    def _merge_segments(segments: List[Segment], max_gap_seconds: float) -> List[Segment]:
        if not segments:
            return []

        segments_sorted = sorted(segments, key=lambda s: (s.start, s.end))
        merged: List[Segment] = []

        for current in segments_sorted:
            if not merged:
                merged.append(current)
                continue

            previous = merged[-1]
            gap = current.start - previous.end

            if gap <= max_gap_seconds:
                merged[-1] = Segment(
                    start=previous.start,
                    end=max(previous.end, current.end),
                )
            else:
                merged.append(current)

        return merged

    @staticmethod
    def _build_masked_speaker_waveform(
        source_waveform: torch.Tensor,
        source_sample_rate: int,
        segments: List[Segment],
    ) -> torch.Tensor:
        total_samples = int(source_waveform.shape[0])
        isolated = torch.zeros_like(source_waveform)

        for seg in segments:
            start_idx = max(0, min(int(seg.start * source_sample_rate), total_samples))
            end_idx = max(0, min(int(seg.end * source_sample_rate), total_samples))
            if end_idx > start_idx:
                isolated[start_idx:end_idx] = source_waveform[start_idx:end_idx]

        return isolated

    @staticmethod
    def _annotation_to_segments_map(diarization) -> Dict[str, List[Segment]]:
        speaker_segments: Dict[str, List[Segment]] = {}

        for turn, _, label in diarization.itertracks(yield_label=True):
            speaker_segments.setdefault(str(label), []).append(
                Segment(start=float(turn.start), end=float(turn.end))
            )

        return speaker_segments

    @staticmethod
    def _chronological_order(speaker_segments: Dict[str, List[Segment]]) -> List[str]:
        first_starts = {
            label: min(seg.start for seg in segments)
            for label, segments in speaker_segments.items()
            if segments
        }
        return sorted(first_starts.keys(), key=lambda label: first_starts[label])

    @classmethod
    def diarize_audio(
        cls,
        audio,
        hf_token: str,
        device: str,
        min_speakers: int,
        max_speakers: int,
        merge_gap_ms: int,
        keep_only_detected_speakers: bool,
    ):
        cls._log("Starting diarization node")

        if min_speakers > max_speakers:
            return cls._silent_outputs(
                audio,
                "Invalid parameters: min_speakers cannot be greater than max_speakers.",
            )

        if not hf_token.strip():
            return cls._silent_outputs(
                audio,
                "Hugging Face token is required to load pyannote diarization pipeline.",
            )

        try:
            sys.setrecursionlimit(3000)
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except RuntimeError as thread_error:
                cls._log(f"Warning while limiting torch threads: {thread_error}")

            processing_device = cls._select_device(device)
            cls._log(f"Using device: {processing_device}")

            original_sample_rate = int(audio["sample_rate"])
            original_waveform = cls._normalize_waveform_shape(audio["waveform"]).detach().cpu()  # type: ignore[index]
            cls._log(
                f"Original waveform shape: {tuple(original_waveform.shape)} @ {original_sample_rate}Hz"
            )

            diar_waveform, diar_sample_rate = cls._resample_if_needed(
                original_waveform,
                original_sample_rate,
            )

            audio_for_diarization = {
                "waveform": diar_waveform.unsqueeze(0),
                "sample_rate": diar_sample_rate,
            }

            cls._log("Loading pyannote pipeline")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            pipeline.to(processing_device)

            cls._log(
                f"Running pipeline with min_speakers={min_speakers}, max_speakers={max_speakers}"
            )
            diarization = pipeline(
                audio_for_diarization,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            speaker_segments = cls._annotation_to_segments_map(diarization)
            if not speaker_segments:
                return cls._silent_outputs(audio, "No speakers detected by diarization pipeline.")

            max_gap_seconds = merge_gap_ms / 1000.0
            merged_segments_map: Dict[str, List[Segment]] = {
                label: cls._merge_segments(segments, max_gap_seconds)
                for label, segments in speaker_segments.items()
            }

            ordered_labels = cls._chronological_order(merged_segments_map)
            cls._log(f"Detected speakers: {ordered_labels}")

            outputs: List[Dict[str, torch.Tensor | int]] = []
            summary_items: List[str] = []
            json_items: List[Dict[str, object]] = []

            for output_index, label in enumerate(ordered_labels[: cls.MAX_OUTPUT_SPEAKERS], start=1):
                segments = merged_segments_map[label]
                isolated_waveform = cls._build_masked_speaker_waveform(
                    source_waveform=original_waveform,
                    source_sample_rate=original_sample_rate,
                    segments=segments,
                )

                if not keep_only_detected_speakers:
                    # Mantido por compatibilidade semântica; o comportamento continua sendo
                    # timeline-preserving com silêncio fora dos segmentos do speaker.
                    pass

                outputs.append(cls._make_audio_output(isolated_waveform, original_sample_rate))

                first_start = min(segment.start for segment in segments)
                total_duration = sum(segment.duration for segment in segments)

                summary_items.append(
                    f"Output {output_index} -> {label} | first_start={first_start:.2f}s | "
                    f"segments={len(segments)} | total_speech={total_duration:.2f}s"
                )

                json_items.append(
                    {
                        "output_index": output_index,
                        "speaker_label": label,
                        "first_start_seconds": cls._round3(first_start),
                        "segment_count": len(segments),
                        "total_speech_seconds": cls._round3(total_duration),
                        "segments": [
                            {
                                "start": cls._round3(segment.start),
                                "end": cls._round3(segment.end),
                                "duration": cls._round3(segment.duration),
                            }
                            for segment in segments
                        ],
                    }
                )

            while len(outputs) < cls.MAX_OUTPUT_SPEAKERS:
                outputs.append(cls._make_silent_output(original_waveform, original_sample_rate))

            summary = "Speakers ordered by first appearance:\n" + "\n".join(summary_items)
            segments_json = json.dumps(
                {
                    "detected_speakers": len(ordered_labels),
                    "returned_outputs": cls.MAX_OUTPUT_SPEAKERS,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                    "merge_gap_ms": merge_gap_ms,
                    "speakers": json_items,
                },
                ensure_ascii=False,
                indent=2,
            )

            return (
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3],
                outputs[4],
                summary,
                segments_json,
            )

        except Exception as exc:
            error_message = (
                f"Error during diarization: {exc}\n{traceback.format_exc()}"
            )
            cls._log(error_message)
            return cls._silent_outputs(audio, error_message)


NODE_CLASS_MAPPINGS = {
    "SpeakerDiarizerChronoNode": SpeakerDiarizerChronoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerDiarizerChronoNode": "Speaker Diarizer Chrono (Up to 5 Speakers)",
}