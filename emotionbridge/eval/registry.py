from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import evaluate


@dataclass(frozen=True, slots=True)
class MetricLoadSpec:
    name: str
    config_name: str | None = None
    revision: str | None = None
    module_type: str | None = None


class HFMetricRegistry:
    def __init__(self, default_revision: str | None = None) -> None:
        self.default_revision = default_revision
        self._cache: dict[tuple[str, str | None, str | None, str | None], Any] = {}

    def load(
        self,
        name: str,
        *,
        config_name: str | None = None,
        revision: str | None = None,
        module_type: str | None = None,
    ) -> Any:
        effective_revision = revision if revision is not None else self.default_revision
        cache_key = (name, config_name, effective_revision, module_type)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        kwargs: dict[str, Any] = {}
        if config_name is not None:
            kwargs["config_name"] = config_name
        if effective_revision is not None:
            kwargs["revision"] = effective_revision
        if module_type is not None:
            kwargs["module_type"] = module_type

        metric = evaluate.load(name, **kwargs)
        self._cache[cache_key] = metric
        return metric

    def load_spec(self, spec: MetricLoadSpec) -> Any:
        return self.load(
            spec.name,
            config_name=spec.config_name,
            revision=spec.revision,
            module_type=spec.module_type,
        )
