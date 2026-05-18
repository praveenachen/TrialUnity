import re


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with", "patients", "patient",
    "study", "trial", "clinical", "condition", "treatment",
}


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def tokenize(value: str | None) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", (value or "").lower())
    return {word for word in words if word not in STOP_WORDS}


def first_sentence(value: str | None, fallback: str = "No summary is available for this trial.") -> str:
    text = normalize_space(value)
    if not text:
        return fallback
    match = re.split(r"(?<=[.!?])\s+", text)
    return match[0][:280]
