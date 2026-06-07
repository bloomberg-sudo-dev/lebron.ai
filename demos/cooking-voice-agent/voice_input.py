"""Voice input using speech_recognition with Google Speech-to-Text."""

from typing import Optional
import speech_recognition as sr

_recognizer = sr.Recognizer()
_recognizer.energy_threshold = 300
_recognizer.dynamic_energy_threshold = True


def listen(timeout: int = 10, phrase_timeout: int = 5) -> Optional[str]:
    """Listen for a voice utterance and return the transcribed text, or None."""
    with sr.Microphone() as source:
        print("[Listening...]")
        try:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
        except sr.WaitTimeoutError:
            return None

    try:
        text = _recognizer.recognize_google(audio)
        print(f"[Heard] {text}")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"[STT error] {e}")
        return None
