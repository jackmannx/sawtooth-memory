"""NER pipeline microbenchmarks."""

from __future__ import annotations

import pytest

from sawtooth_memory.ner import NERPipeline

SAMPLE_TEXT = (
    "Transaction txn_998877_alpha_omega on path /etc/nginx/sites-enabled/api.conf "
    "see https://internal.corp/runbooks/inc-4421 and "
    "arn:aws:s3:us-east-1:123456789012:bucket/prod-data "
    "with uuid 550e8400-e29b-41d4-a716-446655440000."
)


@pytest.mark.benchmark(group="ner")
class TestNERBenchmark:
    def test_extract_typical_message(self, benchmark, ner_pipeline: NERPipeline) -> None:
        benchmark(ner_pipeline.extract, SAMPLE_TEXT)

    def test_extract_large_message(self, benchmark, ner_pipeline: NERPipeline) -> None:
        text = (SAMPLE_TEXT + " filler context. ") * 100
        benchmark(ner_pipeline.extract, text)
