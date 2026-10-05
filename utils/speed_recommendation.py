"""Severity -> recommended safe speed (values come from config.py)."""
import config

DISCLAIMER = ("These speeds are system recommendations for demonstration purposes only. "
              "They are NOT certified road-safety limits; always obey local traffic rules "
              "and use your own judgement.")


def _speed_table() -> dict:
    return {"low": config.LOW_SPEED, "medium": config.MEDIUM_SPEED, "high": config.HIGH_SPEED}


def get_safe_speed(severity: str) -> int:
    try:
        return int(_speed_table()[str(severity).strip().lower()])
    except KeyError:
        raise ValueError(f"Unknown severity '{severity}'. Expected one of {config.SEVERITY_CLASSES}.")


def get_explanation(severity: str) -> str:
    speed = get_safe_speed(severity)
    sev = str(severity).strip().capitalize()
    text = {
        "Low": "Low severity pothole detected. Slow down slightly and stay alert.",
        "Medium": "Medium severity pothole detected. Reduce speed and proceed carefully.",
        "High": "High severity pothole detected. Slow down significantly and avoid the pothole if it is safe to do so.",
    }[sev]
    return f"{text} (Recommended speed: {speed} km/h)"
