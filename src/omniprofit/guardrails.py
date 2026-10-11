"""Guardrails on the advisor's proposals.

A language model can cite numbers that are not in the data or suggest
drastic moves. Each proposed action is checked against the computed report
and a set of fixed limits; anything that fails is rejected with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Action, Advice, GuardedAdvice, ProfitLine, ProfitReport, RejectedAction


@dataclass(frozen=True)
class Limits:
    max_price_rise_pct: float = 10.0
    max_ad_cut_pct: float = 50.0
    # A SKU may only be paused on a channel if it is losing money there.
    pause_requires_negative_contribution: bool = True
    # Never pause a line carrying more than this share of total revenue.
    max_revenue_share_to_pause_pct: float = 15.0
    pct_tolerance: float = 0.5
    currency_tolerance_pct: float = 1.0


def _find(report: ProfitReport, channel: str, sku: str | None) -> ProfitLine | None:
    pool = report.channels if sku is None else report.skus
    return next((line for line in pool if line.channel == channel and line.sku == sku), None)


def check(action: Action, report: ProfitReport, limits: Limits) -> list[str]:
    """Return the reasons this action is unacceptable; an empty list means approved."""
    line = _find(report, action.channel, action.sku)
    if line is None:
        target = action.channel if action.sku is None else f"{action.sku} on {action.channel}"
        return [f"'{target}' does not exist in the data"]

    reasons = []
    actual = getattr(line, action.evidence.metric)
    if action.evidence.metric == "contribution":
        allowed = max(abs(actual) * limits.currency_tolerance_pct / 100, 1.0)
    else:
        allowed = limits.pct_tolerance
    if abs(actual - action.evidence.value) > allowed:
        reasons.append(
            f"cited {action.evidence.metric}={action.evidence.value:g} but the data says {actual:g}"
        )

    if action.action in {"raise_price", "reduce_ad_spend"}:
        cap = limits.max_price_rise_pct if action.action == "raise_price" else limits.max_ad_cut_pct
        if action.change_pct is None or action.change_pct <= 0:
            reasons.append(f"{action.action} needs a positive change_pct")
        elif action.change_pct > cap:
            reasons.append(f"change of {action.change_pct:g}% exceeds the {cap:g}% limit for {action.action}")

    if action.action == "reduce_ad_spend" and line.ad_spend <= 0:
        reasons.append("there is no ad spend to reduce")

    if action.action == "pause_sku_on_channel":
        if action.sku is None:
            reasons.append("pausing needs a specific SKU; whole channels cannot be paused")
        else:
            if limits.pause_requires_negative_contribution and line.contribution >= 0:
                reasons.append(f"line is profitable (contribution {line.contribution:g}); pausing would lose money")
            share = 100 * line.revenue / report.total_revenue if report.total_revenue else 0.0
            if share > limits.max_revenue_share_to_pause_pct:
                reasons.append(
                    f"line carries {share:.1f}% of revenue, above the {limits.max_revenue_share_to_pause_pct:g}% pause limit"
                )
    return reasons


def review(advice: Advice, report: ProfitReport, limits: Limits | None = None) -> GuardedAdvice:
    limits = limits or Limits()
    approved, rejected = [], []
    for action in advice.actions:
        reasons = check(action, report, limits)
        if reasons:
            rejected.append(RejectedAction(action=action, reasons=reasons))
        else:
            approved.append(action)
    return GuardedAdvice(approved=approved, rejected=rejected)
