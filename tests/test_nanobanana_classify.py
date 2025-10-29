"""Tests for NanoBanana failure classification logic."""

from app.services.nanobanana import SafetySummary, classify_failure


def test_classify_failure_detects_blocked_safety() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_VIOLENCE",
                        "blocked": True,
                    }
                ],
                "content": {"parts": [{"text": "blocked"}]},
            }
        ]
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "safety=violence/blocked"
    assert isinstance(summary, SafetySummary)
    assert summary.present is True
    assert summary.triggered is True
    assert summary.categories == ("violence",)
    assert summary.levels == {"violence": "BLOCKED"}


def test_classify_failure_detects_high_probability_safety() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "safetyRatings": [
                    {
                        "category": "IMAGE_VIOLENCE",
                        "probability": "PROBABILITY_MEDIUM",
                    }
                ],
                "content": {"parts": [{"text": "blocked"}]},
            }
        ]
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "safety=image_violence/medium"
    assert summary.present is True
    assert summary.triggered is True
    assert summary.categories == ("image_violence",)
    assert summary.levels == {"image_violence": "MEDIUM"}


def test_classify_failure_marks_transient_for_other_finish_without_safety() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "content": {"parts": [{"text": "no image"}]},
            }
        ]
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "finish=IMAGE_OTHER,no_parts"
    assert summary.present is False
    assert summary.triggered is False
    assert summary.categories == ()
    assert summary.levels == {}


def test_classify_failure_marks_parser_miss_for_alt_refs() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "SUCCESS",
                "content": {
                    "parts": [
                        {
                            "fileUri": "gs://nano-banana/results/example.png",
                        }
                    ]
                },
            }
        ]
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "finish=SUCCESS,alt_refs"
    assert summary.present is False
    assert summary.triggered is False


def test_classify_failure_detects_prompt_feedback_safety() -> None:
    response = {
        "promptFeedback": {
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_VIOLENCE",
                    "likelihood": "LIKELIHOOD_HIGH",
                }
            ]
        },
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "content": {"parts": [{"text": "blocked"}]},
            }
        ],
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "safety=violence/high"
    assert summary.present is True
    assert summary.triggered is True
    assert summary.categories == ("violence",)
    assert summary.levels == {"violence": "HIGH"}


def test_classify_failure_marks_transient_for_low_safety() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_IMAGE_VIOLENCE",
                        "probability": "PROBABILITY_LOW",
                    }
                ],
            }
        ]
    }

    code, detail, summary = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "finish=IMAGE_OTHER,no_parts"
    assert summary.present is True
    assert summary.triggered is False
    assert summary.categories == ("image_violence",)
    assert summary.levels == {"image_violence": "LOW"}
