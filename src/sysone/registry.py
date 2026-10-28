"""Build guarded clients from configs/models.yaml, configs/pricing.yaml and configs/budget.yaml."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import yaml

from sysone.budget.guard import CostGuard
from sysone.budget.ledger import Ledger
from sysone.budget.pricing import PriceTable, UnknownPrice
from sysone.clients.base import GuardedClient, ModelNotConfigured, ModelSpec, Provider
from sysone.clients.cache import ResponseCache

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def _default_provider(name: str) -> Provider:
    if name == "typesafe":
        from sysone.clients.typesafe import TypeSafeProvider

        return TypeSafeProvider()
    if name == "openrouter":
        from sysone.clients.openrouter import OpenRouterProvider

        return OpenRouterProvider()
    if name == "bedrock":
        from sysone.clients.bedrock import BedrockProvider

        return BedrockProvider()
    if name == "gemini":
        from sysone.clients.gemini import GeminiProvider

        # GEMINI_VERTEX_PROJECT routes Gemini through Vertex AI (Gemini Enterprise Agent Platform),
        # billed to that GCP project; otherwise the Gemini API key is used.
        return GeminiProvider(vertex_project=os.environ.get("GEMINI_VERTEX_PROJECT") or None)
    raise ValueError(f"unknown provider '{name}'")


class Registry:
    """One registry per experiment run: a shared cost guard, ledger and cache for all models."""

    def __init__(
        self,
        run_id: str,
        *,
        config_dir: Path = CONFIG_DIR,
        results_dir: Path = RESULTS_DIR,
        provider_factory: Callable[[str], Provider] = _default_provider,
    ):
        self.models = yaml.safe_load((config_dir / "models.yaml").read_text())
        self.prices = PriceTable.load(config_dir / "pricing.yaml")
        budget = yaml.safe_load((config_dir / "budget.yaml").read_text())
        self.max_retries = int(budget["max_retries"])
        self.project_caps_usd = {k: float(v) for k, v in budget["project_caps_usd"].items()}
        self.guard = CostGuard(
            Ledger.for_run(results_dir, run_id),
            per_call_cap_usd=float(budget["per_call_cap_usd"]),
            run_cap_usd=float(budget["run_cap_usd"]),
            project_caps_usd=self.project_caps_usd,
            other_runs_spent_usd=spend_by_provider(results_dir, exclude_run=run_id),
        )
        self.cache = ResponseCache(results_dir / "cache" / "responses.sqlite")
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}

    def spec(self, alias: str) -> ModelSpec:
        entry = self.models[alias]
        provider, model, tier = entry["provider"], entry["model"], entry.get("service_tier")
        try:
            price = self.prices.lookup(provider, model)
        except UnknownPrice:
            price = None
        try:
            tier_price = self.prices.lookup_tier(provider, model, tier) if tier else None
        except UnknownPrice:
            tier_price = None
        return ModelSpec(alias=alias, provider=provider, model=model, price=price, service_tier=tier,
                         reasoning_effort=entry.get("reasoning_effort"), tier_price=tier_price)

    def project_status(self) -> dict[str, dict[str, float]]:
        """Each capped provider's spend across all runs so far, against its project cap."""
        return {
            provider: {"spent_usd": round(self.guard.project_spent_usd(provider), 6), "cap_usd": cap}
            for provider, cap in sorted(self.project_caps_usd.items())
        }

    def client(self, alias: str) -> GuardedClient:
        spec = self.spec(alias)
        if spec.is_placeholder:
            raise ModelNotConfigured(
                f"model alias '{alias}' is a placeholder ({spec.model}); set a real model ID in configs/models.yaml"
            )
        if spec.provider not in self._providers:
            self._providers[spec.provider] = self._provider_factory(spec.provider)
        return GuardedClient(spec, self._providers[spec.provider], self.guard, self.cache, max_retries=self.max_retries)


def spend_by_provider(results_dir: Path, *, exclude_run: str | None = None) -> dict[str, float]:
    """Total cost per provider in every run's spending log under results_dir/runs.

    This is only as complete as results/: a deleted run folder no longer counts, so the
    provider consoles stay the authoritative record of spend.
    """
    totals: dict[str, float] = defaultdict(float)
    for ledger in sorted(Path(results_dir).glob("runs/*/ledger.jsonl")):
        if ledger.parent.name == exclude_run:
            continue
        for line in ledger.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                totals[entry.get("provider", "")] += entry.get("cost_usd", 0.0)
    return dict(totals)
