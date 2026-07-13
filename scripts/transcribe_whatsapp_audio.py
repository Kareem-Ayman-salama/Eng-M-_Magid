"""Transcribe WhatsApp audio files and write a structured markdown report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class AudioInput:
    """Audio input metadata.

    Attributes:
        label: Human-readable file label.
        path: Audio file path.
    """

    label: str
    path: Path


def format_timestamp(seconds: float) -> str:
    """Format seconds as mm:ss.

    Args:
        seconds: Timestamp in seconds.

    Returns:
        Timestamp string.
    """

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def transcribe_files(audio_files: list[AudioInput], output_path: Path) -> None:
    """Transcribe audio files and write markdown output.

    Args:
        audio_files: Audio files to transcribe.
        output_path: Markdown output path.
    """

    model = WhisperModel("small", device="cpu", compute_type="int8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# WhatsApp Audio Transcription",
        "",
        "تفريغ نصي أولي للتسجيلات الصوتية.",
        "",
    ]

    for audio in audio_files:
        lines.extend([f"## {audio.label}", "", f"File: `{audio.path}`", ""])
        segments, info = model.transcribe(
            str(audio.path),
            language="ar",
            beam_size=5,
            vad_filter=True,
        )
        lines.append(f"- Detected language: {info.language}")
        lines.append(f"- Duration: {format_timestamp(info.duration)}")
        lines.append("")
        full_text: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            full_text.append(text)
            lines.append(
                f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {text}"
            )
        lines.extend(["", "**Full text:**", "", " ".join(full_text), ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run transcription for the provided WhatsApp audio files."""

    audio_files = [
        AudioInput(
            label="WhatsApp Audio 2026-07-09 09.25.33",
            path=Path(r"D:\Download google\WhatsApp Audio 2026-07-09 at 9.25.33 AM.ogg"),
        ),
        AudioInput(
            label="WhatsApp Audio 2026-07-09 09.25.33 (1)",
            path=Path(r"D:\Download google\WhatsApp Audio 2026-07-09 at 9.25.33 AM (1).ogg"),
        ),
        AudioInput(
            label="WhatsApp Ptt 2026-07-09 11.56.30",
            path=Path(r"D:\Download google\WhatsApp Ptt 2026-07-09 at 11.56.30 AM.ogg"),
        ),
    ]
    output_path = Path("reports/whatsapp_audio_transcription.md")
    transcribe_files(audio_files, output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
