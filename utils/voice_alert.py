"""Offline voice alerts with pyttsx3 plus a cooldown.

Speech runs in a short-lived subprocess so that it never blocks Streamlit and
avoids the well-known pyttsx3 'run loop already started' problem on reruns.
"""
import subprocess
import sys
import time

import config

_SPEAK_CODE = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)


def build_alert_text(severity: str, speed: int) -> str:
    sev = str(severity).capitalize()
    if sev == "Low":
        return f"Warning. Low severity pothole detected. Recommended speed {speed} kilometers per hour."
    return f"Warning. {sev} severity pothole detected. Reduce speed to {speed} kilometers per hour."


class VoiceAlert:
    def __init__(self, cooldown_sec: float = config.VOICE_COOLDOWN_SEC):
        self.cooldown_sec = cooldown_sec
        self._last_time = 0.0
        self.last_error = None

    def ready(self) -> bool:
        return (time.monotonic() - self._last_time) >= self.cooldown_sec

    def speak(self, text: str, force: bool = False) -> bool:
        """Speak `text`. Returns True if speech was started, False if skipped/failed."""
        if not force and not self.ready():
            return False
        try:
            kwargs = {}
            if sys.platform.startswith("win"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen([sys.executable, "-c", _SPEAK_CODE, text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
            self._last_time = time.monotonic()
            self.last_error = None
            return True
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            return False

    def alert(self, severity: str, speed: int, force: bool = False) -> bool:
        return self.speak(build_alert_text(severity, speed), force=force)
