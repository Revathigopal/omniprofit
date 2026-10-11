"""Runs the agents in sequence and applies the guardrails."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .agents import AdvisorAgent, CostAgent, IngestAgent, ProfitAgent
from .guardrails import Limits, review
from .llm import LLM
from .models import GuardedAdvice, ProfitReport


class RunResult(BaseModel):
    orders_loaded: int
    rejected_rows: list[str]
    report: ProfitReport
    advice: GuardedAdvice | None = None


class Orchestrator:
    def __init__(self, llm: LLM | None = None, limits: Limits | None = None) -> None:
        self.llm = llm
        self.limits = limits or Limits()

    def run(self, data_dir: Path, advise: bool = True) -> RunResult:
        dataset = IngestAgent().run(data_dir)
        economics = CostAgent().run(dataset)
        report = ProfitAgent().run(economics, dataset.ad_spend)
        advice = None
        if advise:
            if self.llm is None:
                raise ValueError("an LLM is required to produce advice")
            advice = review(AdvisorAgent(self.llm).run(report), report, self.limits)
        return RunResult(
            orders_loaded=len(dataset.orders),
            rejected_rows=dataset.rejected_rows,
            report=report,
            advice=advice,
        )
