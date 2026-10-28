"""Build guarded clients from configs/models.yaml, configs/pricing.yaml and configs/budget.yaml."""

from __future__ import annotations

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

        return GeminiProvider()
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
        self.guard = CostGuard(
            Ledger.for_run(results_dir, run_id),
            per_call_cap_usd=float(budget["per_call_cap_usd"]),
            run_cap_usd=float(budget["run_cap_usd"]),
        )
        self.cache = ResponseCache(results_dir / "cache" / "responses.sqlite")
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}

    def spec(self, alias: str) -> ModelSpec:
        entry = self.models[alias]
        try:
            price = self.prices.lookup(entry["provider"], entry["model"])
        except UnknownPrice:
            price = None
        return ModelSpec(alias=alias, provider=entry["provider"], model=entry["model"], price=price)

    def client(self, alias: str) -> GuardedClient:
        spec = self.spec(alias)
        if spec.is_placeholder:
            raise ModelNotConfigured(
                f"model alias '{alias}' is a placeholder ({spec.model}); set a real model ID in configs/models.yaml"
            )
        if spec.provider not in self._providers:
            self._providers[spec.provider] = self._provider_factory(spec.provider)
        return GuardedClient(spec, self._providers[spec.provider], self.guard, self.cache, max_retries=self.max_retries)
