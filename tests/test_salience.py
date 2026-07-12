"""Tests for the salience heuristic entity extractor."""

from sawtooth_memory.salience import (
    SalienceConfig,
    SalienceEntityExtractor,
    score_candidate,
)


class TestScoreCandidate:
    def test_high_salience_near_cue_word(self):
        text = "Please use tracking code ALPHA-991 for this shipment."
        score, key = score_candidate("ALPHA-991", text)
        assert score >= 0.5
        assert key == "tracking_code"

    def test_incident_ticket_format(self):
        text = "Open ticket INC-4421 needs escalation."
        score, key = score_candidate("INC-4421", text)
        assert score >= 0.5
        assert key == "ticket_id"

    def test_common_word_rejected(self):
        score, _ = score_candidate("quantum", "Explain quantum physics.")
        assert score < 0.5

    def test_pure_number_rejected(self):
        score, _ = score_candidate("12345", "The answer is 12345.")
        assert score < 0.5


class TestSalienceEntityExtractor:
    def test_extracts_unstructured_tracking_code(self):
        extractor = SalienceEntityExtractor()
        text = "Ship with tracking code ALPHA-991 by Friday."
        result = extractor.extract(text)
        assert any(v == "ALPHA-991" for v in result.values())

    def test_extracts_incident_id(self):
        extractor = SalienceEntityExtractor()
        text = "Escalate incident INC-4421 to on-call."
        result = extractor.extract(text)
        assert any(v == "INC-4421" for v in result.values())

    def test_extracts_colon_separated_ref(self):
        extractor = SalienceEntityExtractor()
        text = "Customer ref: JSMITH-2024 was approved."
        result = extractor.extract(text)
        assert any(v == "JSMITH-2024" for v in result.values())

    def test_excludes_already_captured_values(self):
        extractor = SalienceEntityExtractor()
        text = "Tracking code ALPHA-991 confirmed."
        result = extractor.extract(text, exclude_values=["ALPHA-991"])
        assert "ALPHA-991" not in result.values()

    def test_respects_threshold(self):
        extractor = SalienceEntityExtractor(
            SalienceConfig(threshold=0.99, max_entities=10)
        )
        text = "Some loosely formatted value abc."
        result = extractor.extract(text)
        assert result == {}

    def test_respects_max_entities(self):
        extractor = SalienceEntityExtractor(
            SalienceConfig(threshold=0.3, max_entities=2)
        )
        text = (
            "ticket INC-001, ticket INC-002, ticket INC-003, "
            "tracking code ALPHA-991"
        )
        result = extractor.extract(text)
        assert len(result) <= 2
