import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEEDBACK_FILE = (
    PROJECT_ROOT
    / "feedback"
    / "risk_overrides.json"
)


def load_overrides():
    """Load all previously saved risk overrides."""

    if not FEEDBACK_FILE.exists():
        return []

    try:
        with open(
            FEEDBACK_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


def save_risk_override(
    report_id,
    report_text,
    original_risk,
    corrected_risk,
    reason
):
    """
    Store a safety officer's correction to an AI risk decision.
    """

    valid_levels = {"LOW", "MEDIUM", "HIGH"}

    original_risk = original_risk.upper()
    corrected_risk = corrected_risk.upper()

    if corrected_risk not in valid_levels:
        raise ValueError(
            "Corrected risk must be LOW, MEDIUM, or HIGH."
        )

    if not reason.strip():
        raise ValueError(
            "A reason is required for a risk override."
        )

    overrides = load_overrides()

    correction = {
        "report_id": str(report_id),
        "report_text": report_text,
        "original_risk": original_risk,
        "corrected_risk": corrected_risk,
        "reason": reason.strip(),
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        )
    }

    overrides.append(correction)

    FEEDBACK_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FEEDBACK_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            overrides,
            file,
            indent=2,
            ensure_ascii=False
        )

    return correction


def get_recent_overrides(limit=5):
    """
    Return the latest human corrections for few-shot learning.
    """

    overrides = load_overrides()

    return overrides[-limit:]


def format_overrides_for_prompt(limit=5):
    """
    Convert recent human corrections into examples that can
    be inserted into the Risk Scoring Agent prompt.
    """

    overrides = get_recent_overrides(limit)

    if not overrides:
        return (
            "No previous safety officer corrections "
            "are available."
        )

    examples = []

    for number, item in enumerate(
        overrides,
        start=1
    ):
        example = f"""
Example {number}

Safety report:
{item["report_text"]}

Original AI classification:
{item["original_risk"]}

Safety officer classification:
{item["corrected_risk"]}

Safety officer reason:
{item["reason"]}
"""
        examples.append(example.strip())

    return "\n\n".join(examples)