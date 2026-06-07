"""Text-to-speech output using edge-tts (Microsoft Azure Neural TTS)."""

import asyncio
import os
import platform
import subprocess
import tempfile

import edge_tts

VOICE = "en-US-AriaNeural"


async def _synthesize(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)


def _play_audio(path: str):
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["afplay", path], check=True)
    elif system == "Windows":
        subprocess.run(
            ["powershell", "-c", f'(New-Object Media.SoundPlayer "{path}").PlaySync()'],
            check=True,
        )
    else:
        # Linux — try mpg123, fall back to ffplay
        for player in ("mpg123", "ffplay"):
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                args = [player, path] if player == "mpg123" else [player, "-nodisp", "-autoexit", path]
                subprocess.run(args, check=True)
                return
        print(f"[TTS] No audio player found. Install mpg123 or ffplay to hear Chef Buddy.")


def speak(text: str):
    """Synthesize and play text aloud. Blocks until playback finishes."""
    if not text.strip():
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        asyncio.run(_synthesize(text, tmp_path))
        _play_audio(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
