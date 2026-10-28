"""Checks against the real config files, with a fake provider standing in for the network."""

import pytest

from sysone.clients.base import ModelNotConfigured
from sysone.clients.fake import FakeProvider
from sysone.registry import CONFIG_DIR, Registry
from sysone.types import ChatRequest


@pytest.fixture
def registry(tmp_path):
    return Registry("test-run", results_dir=tmp_path, provider_factory=lambda name: FakeProvider())


def test_budget_config_values(registry):
    assert registry.guard.per_call_cap_usd == 2.0
    assert registry.guard.run_cap_usd == 25.0


def test_every_model_is_real_and_priced(registry):
    for alias in registry.models:
        spec = registry.spec(alias)
        assert not spec.is_placeholder, alias
        assert spec.price is not None, alias
        if spec.service_tier:
            assert spec.tier_price is not None, alias
            # A discount tier never costs more than standard.
            assert spec.tier_price.input_per_m <= spec.price.input_per_m, alias
            assert spec.tier_price.output_per_m <= spec.price.output_per_m, alias


def test_model_choices(registry):
    expected = {
        "case_writer": ("bedrock", "qwen.qwen3-235b-a22b-2507-v1:0"),
        "case_writer_2": ("bedrock", "mistral.mistral-large-3-675b-instruct"),
        "explainer_bedrock": ("bedrock", "global.anthropic.claude-haiku-5-5"),
        "explainer_gemini": ("gemini", "gemini-3.1-flash-lite"),
        "explainer_open": ("bedrock", "us.meta.llama4-maverick-17b-instruct-v1:0"),
        "simulator": ("bedrock", "openai.gpt-oss-120b-1:0"),
        "judge": ("bedrock", "openai.gpt-oss-120b-1:0"),
        "decision_alt": ("bedrock", "amazon.nova-lite-v1:0"),
    }
    for alias, (provider, model) in expected.items():
        spec = registry.spec(alias)
        assert (spec.provider, spec.model) == (provider, model), alias
    assert registry.spec("simulator").service_tier == "flex"
    assert registry.spec("simulator").reasoning_effort == "low"
    assert registry.spec("explainer_gemini").reasoning_effort == "minimal"


def test_roles_keep_model_families_apart(registry):
    from sysone.experiments.pilot1 import model_family

    family = {alias: model_family(registry.spec(alias)) for alias in registry.models}
    explainers = {family[a] for a in ("explainer_bedrock", "explainer_gemini", "explainer_open")}
    assert len(explainers) == 3
    for scorer in ("simulator", "judge"):
        assert family[scorer] not in explainers | {family["jev"]}
    for writer in ("case_writer", "case_writer_2"):
        assert family[writer] not in explainers | {family["jev"]}
    assert family["case_writer"] != family["case_writer_2"]


def test_every_used_provider_has_a_project_cap(registry):
    assert {registry.spec(a).provider for a in registry.models} <= set(registry.project_caps_usd)
    assert registry.project_caps_usd == {"bedrock": 50.0, "gemini": 50.0, "typesafe": 5.0, "openrouter": 0.0}


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


def test_placeholder_models_refuse_to_run(tmp_path):
    config = tmp_path / "configs"
    config.mkdir()
    for name in ("pricing.yaml", "budget.yaml"):
        (config / name).write_text((CONFIG_DIR / name).read_text())
    (config / "models.yaml").write_text("someday:\n  provider: bedrock\n  model: TODO_MODEL_ID\n")
    built = []
    registry = Registry("test-run", config_dir=config, results_dir=tmp_path,
                        provider_factory=lambda name: built.append(name) or FakeProvider())
    with pytest.raises(ModelNotConfigured):
        registry.client("someday").complete(ChatRequest.from_prompt("x", max_output_tokens=10))
    assert built == []  # refused before any provider (or credential) is touched


def test_project_spend_is_summed_across_runs(tmp_path):
    from sysone.budget import Ledger

    Ledger.for_run(tmp_path, "earlier").append(provider="bedrock", cost_usd=1.25)
    Ledger.for_run(tmp_path, "earlier").append(provider="typesafe", cost_usd=0.5)
    Ledger.for_run(tmp_path, "other").append(provider="bedrock", cost_usd=0.25)
    registry = Registry("now", results_dir=tmp_path, provider_factory=lambda name: FakeProvider())
    status = registry.project_status()
    assert status["bedrock"] == {"spent_usd": 1.5, "cap_usd": 50.0}
    assert status["typesafe"]["spent_usd"] == 0.5
