"""One local Williams AI: Qwen (+ optional LoRA), books and durable memory.

All cognitive calls use the same effective model. Facts retain their source
and time, generated notes stay unverified, and an independent critic reviews
decisions. This interface has no order executor or broker credentials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .book_index import _private_database
from .model_bundle import runtime_identity
from .unified_memory import UnifiedMemory


class WilliamsAI:
    def __init__(self, *, local_model: Path, memory: Path,
                 book_index: Path | None = None, public_rules_only: bool = False,
                 lora_adapter: Path | None = None, lora_manifest: Path | None = None,
                 n_ctx: int = 8192, reasoning_budget: int = 1536,
                 loader: Callable[..., Any] | None = None):
        if (type(public_rules_only) is not bool or
                (book_index is None) != public_rules_only):
            raise ValueError("CHOOSE_ORIGINAL_BOOK_INDEX_OR_EXPLICIT_PUBLIC_RULE_MODE")
        self.book_index = _private_database(book_index) if book_index is not None else None
        if self.book_index is not None and self.book_index == _private_database(memory):
            raise ValueError("ORIGINAL_BOOK_INDEX_AND_LEARNING_MEMORY_MUST_BE_SEPARATE")
        self.identity = runtime_identity(local_model, adapter=lora_adapter, manifest=lora_manifest)
        self.memory = UnifiedMemory(memory)
        self.public_rules_only = public_rules_only
        self._model = None
        self._model_path = Path(local_model)
        self._runtime_options = {"n_ctx": n_ctx, "reasoning_budget": reasoning_budget,
            "lora_adapter": lora_adapter, "lora_manifest": lora_manifest,
            "use_decision_schema": True, "learning_mode": "auto", "loader": loader}

    @property
    def model(self):
        if self._model is None:
            from .local_model import LocalWilliamsModel
            model = LocalWilliamsModel(self._model_path, **self._runtime_options)
            if model.model_id != self.identity["model_id"]:
                raise ValueError("EFFECTIVE_MODEL_CHANGED_SINCE_MEMORY_BINDING")
            architecture = getattr(model._model, "metadata", {}).get("general.architecture", "")
            if not isinstance(architecture, str) or not architecture.startswith("qwen"):
                raise ValueError("UNIFIED_WILLIAMS_AI_REQUIRES_QWEN_RUNTIME")
            self._model = model
        return self._model

    def study(self, *, editions: tuple[str, ...] = ("TC1", "NTD", "TC2", "COURSE"),
              budget_per_book: int = 2, rounds: int = 1) -> dict[str, Any]:
        if self.book_index is None:
            raise ValueError("ORIGINAL_PRIVATE_BOOKS_REQUIRED_FOR_FULL_SOURCE_STUDY")
        from .study_cli import train_private_curriculum
        return train_private_curriculum(self.book_index, self.memory.path, self.model,
            model_id=self.identity["model_id"], editions=editions,
            budget_per_book=budget_per_book, rounds=rounds)

    def practice(self, *, max_cases: int = 8) -> dict[str, Any]:
        from .adaptive_coach import coach_practice
        return coach_practice(self.model, self.memory.path,
            model_id=self.identity["model_id"], max_cases=max_cases)

    def analyze(self, symbol: str, rows: list[dict[str, Any]], *, tick_size: float,
                bar_index: int, edition: str = "TC2", question: str | None = None,
                campaign_journal: dict[str, Any] | None = None,
                include_topdown: bool = True) -> dict[str, Any]:
        from .reasoning_trial import evaluate_historical_thought
        result = evaluate_historical_thought(symbol, rows, tick_size=tick_size,
            bar_index=bar_index, edition=edition, model=self.model,
            book_index=self.book_index, model_id=self.identity["model_id"],
            study_memory=self.memory.path, coach_notebook=self.memory.path,
            experience_journal=self.memory.path if edition == "TC2" else None,
            research_memory=self.memory.path, research_question=question,
            public_rule_learning=self.public_rules_only,
            campaign_journal=campaign_journal, include_topdown=include_topdown)
        result["unified_ai"] = {"effective_model": self.model.runtime_identity,
            "one_runtime_for_study_practice_and_chart_analysis": True,
            "one_durable_learning_memory": True,
            "memory_counts": self.memory.inventory(),
            "all_knowledge_verified": False, "execution_allowed": False, "trade_action": None}
        return result

    def status(self) -> dict[str, Any]:
        from .study import study_report
        books = (study_report(self.book_index, self.memory.path, model_id=self.identity["model_id"])
                 if self.book_index is not None else None)
        return {"mode": "ONE_LOCAL_WILLIAMS_AI_WITH_SOURCE_AND_EXPERIENCE_MEMORY",
            "effective_model": self.model.runtime_identity if self._model is not None else self.identity,
            "runtime_loaded": self._model is not None,
            "source_mode": "PUBLIC_PARAPHRASES_ONLY" if self.public_rules_only else "PRIVATE_ORIGINAL_BOOK_INDEX",
            "book_study": books, "memory_counts": self.memory.inventory(),
            "adapter_activation_is_not_qualification": True,
            "autonomous_trading_qualified": False, "execution_allowed": False, "trade_action": None}


def main() -> None:
    p = argparse.ArgumentParser(description="One Williams AI runtime and persistent private memory")
    p.add_argument("--local-model", type=Path, required=True)
    p.add_argument("--memory", type=Path, required=True)
    p.add_argument("--book-index", type=Path)
    p.add_argument("--public-rule-learning", action="store_true")
    p.add_argument("--lora-adapter", type=Path)
    p.add_argument("--lora-manifest", type=Path)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--reasoning-budget", type=int, default=1536)
    actions = p.add_subparsers(dest="action", required=True)
    actions.add_parser("status")
    imported = actions.add_parser("import-memory")
    imported.add_argument("checkpoint", type=Path)
    retained = actions.add_parser("retain-material")
    retained.add_argument("file", type=Path)
    retained.add_argument("--source", required=True)
    retained.add_argument("--title", required=True)
    retained.add_argument("--edition", choices=["TC1", "NTD", "TC2", "COURSE", "GENERAL"], required=True)
    retained.add_argument("--available-at", required=True)
    study = actions.add_parser("study")
    study.add_argument("--budget-per-book", type=int, default=2)
    study.add_argument("--rounds", type=int, default=1)
    practice = actions.add_parser("practice")
    practice.add_argument("--cases", type=int, default=8)
    analysis = actions.add_parser("analyze")
    analysis.add_argument("--symbol", required=True)
    analysis.add_argument("--bar-index", type=int, required=True)
    analysis.add_argument("--edition", choices=["TC1", "NTD", "TC2", "COURSE"], default="TC2")
    analysis.add_argument("--question")
    analysis.add_argument("--no-topdown", action="store_true")
    a = p.parse_args()
    brain = WilliamsAI(local_model=a.local_model, memory=a.memory, book_index=a.book_index,
        public_rules_only=a.public_rule_learning, lora_adapter=a.lora_adapter,
        lora_manifest=a.lora_manifest, n_ctx=a.ctx, reasoning_budget=a.reasoning_budget)
    if a.action == "status":
        result = brain.status()
    elif a.action == "import-memory":
        result = brain.memory.import_checkpoint(a.checkpoint)
    elif a.action == "retain-material":
        from .private_lora import _private_file
        path = _private_file(a.file)
        if path.stat().st_size > 2_000_000:
            raise ValueError("BOUNDED_PRIVATE_UTF8_MATERIAL_REQUIRED")
        result = {"retained_document_sha256": brain.memory.retain(path.read_text(encoding="utf-8"),
            source=a.source, title=a.title, edition=a.edition, available_at=a.available_at)}
    elif a.action == "study":
        result = brain.study(budget_per_book=a.budget_per_book, rounds=a.rounds)
    elif a.action == "practice":
        result = brain.practice(max_cases=a.cases)
    else:
        from .causal_paper_lab import frozen_markets
        market = frozen_markets()[a.symbol]
        result = brain.analyze(a.symbol, market["bars"], tick_size=market["tick_size"],
            bar_index=a.bar_index, edition=a.edition, question=a.question,
            include_topdown=not a.no_topdown)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
