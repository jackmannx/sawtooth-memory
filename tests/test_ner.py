"""Tests for the NER pipeline and regex multi-match extraction."""

from sawtooth_memory.ner import NERPipeline, RegexEntityExtractor


class TestRegexEntityExtractor:
    def test_extracts_single_uuid(self):
        extractor = RegexEntityExtractor()
        text = "User id is 550e8400-e29b-41d4-a716-446655440000."
        result = extractor.extract(text)
        assert result["uuid"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_extracts_multiple_uuids(self):
        extractor = RegexEntityExtractor()
        text = (
            "First 550e8400-e29b-41d4-a716-446655440000 and "
            "second 6ba7b810-9dad-11d1-80b4-00c04fd430c8."
        )
        result = extractor.extract(text)
        assert result["uuid"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["uuid_2"] == "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

    def test_custom_pattern(self):
        extractor = RegexEntityExtractor(
            {"transaction_id": r"txn_[a-z0-9_]+"}
        )
        text = "Payment txn_998877_alpha_omega completed."
        result = extractor.extract(text)
        assert result["transaction_id"] == "txn_998877_alpha_omega"


class TestNERPipeline:
    def test_combines_regex_and_salience(self):
        pipeline = NERPipeline.from_config(
            enable=True,
            custom_patterns={"transaction_id": r"txn_[a-z0-9_]+"},
            enable_salience=True,
            salience_threshold=0.4,
        )
        text = (
            "Payment txn_998877_alpha_omega done. "
            "Escalate ticket INC-4421 immediately."
        )
        result = pipeline.extract_with_metadata(text)
        assert "transaction_id" in result.entities
        assert result.strategies["transaction_id"] == "deterministic"
        assert any(v == "INC-4421" for v in result.entities.values())
        salience_keys = [
            k for k, s in result.strategies.items() if s == "salience_heuristic"
        ]
        assert len(salience_keys) >= 1

    def test_salience_skips_regex_captured_values(self):
        pipeline = NERPipeline.from_config(enable=True, enable_salience=True)
        text = "txn_998877_alpha_omega is the transaction id."
        result = pipeline.extract_with_metadata(text)
        values = list(result.entities.values())
        assert values.count("txn_998877_alpha_omega") == 1

    def test_disabled_returns_empty(self):
        pipeline = NERPipeline.from_config(enable=False)
        result = pipeline.extract_with_metadata("ticket INC-4421")
        assert result.entities == {}

    def test_backward_compatible_extract(self):
        pipeline = NERPipeline.from_config(enable=True)
        text = "Path /etc/nginx/api.conf exists."
        entities = pipeline.extract(text)
        assert "file_path" in entities
