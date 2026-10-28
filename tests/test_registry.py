"""Checks against the real config files, with a fake provider standing in for the network."""

import pytest

from sysone.clients.base import ModelNotConfigured
from sysone.clients.fake import FakeProvider
from sysone.registry import Registry
from sysone.types import ChatRequest


@pytest.fixture
def registry(tmp_path):
    return Registry("test-run", results_dir=tmp_path, provider_factory=lambda name: FakeProvider())


def test_budget_config_values(registry):
    assert registry.guard.per_call_cap_usd == 2.0
    assert registry.guard.run_cap_usd == 25.0


def test_case_writer_is_opus_45_on_bedrock_and_priced(registry):
    spec = registry.spec("case_writer")
    assert spec.provider == "bedrock"
    assert spec.model == "us.anthropic.claude-opus-4-5-20251101-v1:0"
    assert spec.price is not None


def test_jev_is_typesafe_latest_and_priced(registry):
    spec = registry.spec("jev")
    assert spec.provider == "typesafe" and spec.model == "jev-latest"
    # Input tokens only; output is free.
    assert spec.price is not None and spec.price.input_per_m > 0 and spec.price.output_per_m == 0


def test_jev_client_builds_the_typesafe_provider(tmp_path):
    built = []
    registry = Registry("test-run", results_dir=tmp_path, provider_factory=lambda name: built.append(name) or FakeProvider())
    registry.client("jev")
    assert built == ["typesafe"]


@pytest.mark.parametrize("alias", ["explainer_bedrock", "explainer_gemini", "judge"])
def test_todo_aliases_refuse_to_run(tmp_path, alias):
    built = []
    registry = Registry("test-run", results_dir=tmp_path, provider_factory=lambda name: built.append(name) or FakeProvider())
    with pytest.raises(ModelNotConfigured):
        registry.client(alias).complete(ChatRequest.from_prompt("x", max_output_tokens=10))
    assert built == []  # refused before any provider (or credential) is touched
