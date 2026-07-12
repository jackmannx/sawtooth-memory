"""Tests for entity guard merge, verifier, and protection manifest."""

from sawtooth_memory.entity_guard import (
    apply_entity_guard,
    build_compression_user_content,
    format_protection_manifest,
    secure_merge_entities,
    verify_protected_entities,
)
from sawtooth_memory.ner import ExtractionResult


class TestSecureMerge:
    def test_local_overrides_llm(self):
        llm = {"uuid": "hallucinated-id"}
        local = {"uuid": "550e8400-e29b-41d4-a716-446655440000"}
        merged = secure_merge_entities(llm, local)
        assert merged["uuid"] == "550e8400-e29b-41d4-a716-446655440000"


class TestVerifier:
    def test_reinjects_dropped_entity(self):
        protected = {"ticket_id": "INC-4421"}
        combined = {"other_key": "value"}
        narrative = "The incident was escalated."
        reinjected = verify_protected_entities(protected, combined, narrative)
        assert reinjected == {"ticket_id": "INC-4421"}

    def test_skips_entity_in_narrative(self):
        protected = {"ticket_id": "INC-4421"}
        combined = {}
        narrative = "Escalated INC-4421 to on-call."
        reinjected = verify_protected_entities(protected, combined, narrative)
        assert reinjected == {}

    def test_skips_entity_already_present(self):
        protected = {"ticket_id": "INC-4421"}
        combined = {"ticket_id": "INC-4421"}
        reinjected = verify_protected_entities(protected, combined, "")
        assert reinjected == {}


class TestProtectionManifest:
    def test_format_manifest(self):
        manifest = format_protection_manifest(
            {"ticket_id": "INC-4421", "uuid": "abc-123"}
        )
        assert "PROTECTED VALUES" in manifest
        assert "ticket_id: INC-4421" in manifest
        assert "uuid: abc-123" in manifest

    def test_empty_manifest(self):
        assert format_protection_manifest({}) == ""

    def test_user_content_includes_manifest(self):
        content = build_compression_user_content(
            "USER: hello",
            {"ticket_id": "INC-4421"},
        )
        assert "PROTECTED VALUES" in content
        assert "USER: hello" in content


class TestApplyEntityGuard:
    def test_full_pipeline(self):
        extraction = ExtractionResult(
            entities={"ticket_id": "INC-4421"},
            strategies={"ticket_id": "salience_heuristic"},
        )
        llm_entities = {"summary_note": "escalated"}
        narrative = "The ticket was escalated."

        combined, strategies = apply_entity_guard(
            extraction,
            llm_entities,
            narrative,
            enable_verifier=True,
        )
        assert combined["ticket_id"] == "INC-4421"
        assert combined["summary_note"] == "escalated"
        assert strategies["ticket_id"] == "salience_heuristic"
        assert strategies["summary_note"] == "llm_synthesis"

    def test_pinned_entities_tagged(self):
        extraction = ExtractionResult()
        combined, strategies = apply_entity_guard(
            extraction,
            {},
            "",
            pinned_entities={"api_key": "sk-test"},
            enable_verifier=False,
        )
        assert combined["api_key"] == "sk-test"
        assert strategies["api_key"] == "pinned"
