"""Tests for NanoBanana failure classification logic."""

from app.services.nanobanana import classify_failure


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

    code, detail = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "safety=violence/blocked"


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

    code, detail = classify_failure(response)
    assert code == "UNSUITABLE_PHOTO"
    assert detail == "safety=image_violence/medium"


def test_classify_failure_marks_transient_for_other_finish_without_safety() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "content": {"parts": [{"text": "no image"}]},
            }
        ]
    }

    code, detail = classify_failure(response)
    assert code == "TRANSIENT"
    assert detail == "finish=IMAGE_OTHER,no_parts"


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

    code, detail = classify_failure(response)
    assert code == "PARSER_MISS"
    assert detail == "finish=SUCCESS,alt_refs"
