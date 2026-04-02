from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchaudio
from comfy import model_management as mm


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class SpeakerDiarizerChronoNode:
    """
    ComfyUI node that isolates up to 5 speakers in chronological order
    using an external pyannote.audio 4.x worker.

    Output mapping:
    - speaker_1_audio = first unique speaker to speak
    - speaker_2_audio = second unique speaker to speak
    - ...
    - speaker_5_audio = fifth unique speaker to speak
    """

    MAX_OUTPUT_SPEAKERS = 5

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
                        "tooltip": "Hugging Face token for pyannote community-1",
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Device used by the external pyannote worker",
                    },
                ),
                "min_speakers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Minimum expected speakers",
                    },
                ),
                "max_speakers": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Maximum expected speakers",
                    },
                ),
                "merge_gap_ms": (
                    "INT",
                    {
                        "default": 200,
                        "min": 0,
                        "max": 5000,
                        "step": 50,
                        "tooltip": "Merge adjacent same-speaker segments up to this gap",
                    },
                ),
                "keep_only_detected_speakers": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Kept for compatibility. Output is timeline-preserving.",
                    },
                ),
                "pyannote_python": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path to Python executable of the external pyannote environment",
                    },
                ),
                "worker_script": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path to pyannote_worker.py",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 600,
                        "min": 30,
                        "max": 7200,
                        "step": 30,
                        "tooltip": "Timeout for the external worker",
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
        return int(float(value) * 1000 + 0.5) / 1000

    @staticmethod
    def _normalize_waveform_shape(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:  # (B, C, S)
            return waveform[0].mean(dim=0)
        if waveform.ndim == 2:  # (C, S)
            return waveform.mean(dim=0)
        if waveform.ndim == 1:  # (S,)
            return waveform
        raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")

    @classmethod
    def _make_audio_output(
        cls,
        waveform_1d: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, torch.Tensor | int]:
        return {
            "waveform": waveform_1d.unsqueeze(0).unsqueeze(0),
            "sample_rate": sample_rate,
        }

    @classmethod
    def _make_silent_output(
        cls,
        reference_waveform_1d: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, torch.Tensor | int]:
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
    def _select_device(device: str) -> str:
        if device == "auto":
            torch_device = mm.get_torch_device()
            return "cuda" if str(torch_device).startswith("cuda") else "cpu"

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA foi solicitado, mas não está disponível.")
            return "cuda"

        return "cpu"

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
    def _validate_external_paths(pyannote_python: str, worker_script: str) -> None:
        if not pyannote_python.strip():
            raise ValueError("O campo pyannote_python está vazio.")
        if not worker_script.strip():
            raise ValueError("O campo worker_script está vazio.")
        if not os.path.isfile(pyannote_python):
            raise FileNotFoundError(f"Python externo não encontrado: {pyannote_python}")
        if not os.path.isfile(worker_script):
            raise FileNotFoundError(f"Worker script não encontrado: {worker_script}")

    @classmethod
    def _call_external_worker(
        cls,
        wav_path: str,
        json_path: str,
        pyannote_python: str,
        worker_script: str,
        hf_token: str,
        device: str,
        min_speakers: int,
        max_speakers: int,
        timeout_seconds: int,
    ) -> Dict[str, object]:
        cls._validate_external_paths(pyannote_python, worker_script)

        command = [
            pyannote_python,
            worker_script,
            "--input-wav",
            wav_path,
            "--output-json",
            json_path,
            "--hf-token",
            hf_token,
            "--device",
            device,
            "--min-speakers",
            str(min_speakers),
            "--max-speakers",
            str(max_speakers),
        ]

        cls._log("Executing external pyannote worker")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        if completed.stdout:
            cls._log("Worker stdout:\n" + completed.stdout)
        if completed.stderr:
            cls._log("Worker stderr:\n" + completed.stderr)

        if completed.returncode != 0:
            raise RuntimeError(
                f"External worker failed with exit code {completed.returncode}.\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Worker did not create output JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        return payload

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
        pyannote_python: str,
        worker_script: str,
        timeout_seconds: int,
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

            external_device = cls._select_device(device)
            cls._log(f"Using external worker device: {external_device}")

            original_sample_rate = int(audio["sample_rate"])
            original_waveform = cls._normalize_waveform_shape(audio["waveform"]).detach().cpu()  # type: ignore[index]
            cls._log(
                f"Original waveform shape: {tuple(original_waveform.shape)} @ {original_sample_rate}Hz"
            )

            with tempfile.TemporaryDirectory(prefix="comfyui_pyannote_") as temp_dir:
                wav_path = os.path.join(temp_dir, "input.wav")
                json_path = os.path.join(temp_dir, "segments.json")

                torchaudio.save(
                    wav_path,
                    original_waveform.unsqueeze(0).to(torch.float32),
                    sample_rate=original_sample_rate,
                )
                cls._log(f"Temporary WAV written to: {wav_path}")

                payload = cls._call_external_worker(
                    wav_path=wav_path,
                    json_path=json_path,
                    pyannote_python=pyannote_python,
                    worker_script=worker_script,
                    hf_token=hf_token,
                    device=external_device,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    timeout_seconds=timeout_seconds,
                )

            raw_speakers = payload.get("speakers", [])
            if not isinstance(raw_speakers, list) or not raw_speakers:
                return cls._silent_outputs(
                    audio,
                    "No speakers detected by external pyannote worker.",
                )

            speakers_to_process: list[dict[str, object]] = list(
                itertools.islice(
                    (s for s in raw_speakers if isinstance(s, dict)),
                    cls.MAX_OUTPUT_SPEAKERS,
                )
            )

            max_gap_seconds = merge_gap_ms / 1000.0
            outputs: list[dict[str, torch.Tensor | int]] = []
            summary_items: list[str] = []
            json_items: list[dict[str, object]] = []

            for output_index, speaker_data in enumerate(speakers_to_process, start=1):
                label = str(speaker_data.get("speaker_label", f"SPEAKER_{output_index:02d}"))

                raw_segments = speaker_data.get("segments", [])
                if not isinstance(raw_segments, list):
                    raw_segments = []

                segments = [
                    Segment(
                        start=float(segment["start"]),
                        end=float(segment["end"]),
                    )
                    for segment in raw_segments
                    if isinstance(segment, dict)
                    and "start" in segment
                    and "end" in segment
                ]

                segments: List[Segment] = cls._merge_segments(segments, max_gap_seconds)

                isolated_waveform = cls._build_masked_speaker_waveform(
                    source_waveform=original_waveform,
                    source_sample_rate=original_sample_rate,
                    segments=segments,
                )

                if not keep_only_detected_speakers:
                    pass

                outputs.append(cls._make_audio_output(isolated_waveform, original_sample_rate))

                first_start = min((segment.start for segment in segments), default=0.0)
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
                    "detected_speakers": len(raw_speakers),
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
    "SpeakerDiarizerChronoNode": "Speaker Diarizer (Up to 5 Speakers)",
}