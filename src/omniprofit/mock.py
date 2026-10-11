"""Offline rule-based stand-in for the advisor's language model.

This is NOT a model. It applies fixed thresholds to the profit report so the
system runs and is testable with no network. Use a real provider for advice
that reasons about the figures.
"""

from __future__ import annotations

import json

from .llm import Message


def mock_handler(messages: list[Message]) -> str:
    report = json.loads(messages[1]["content"])
    actions = []
    for line in report["channels"]:
        if line["ad_share_pct"] > 12:
            actions.append(
                {
                    "action": "reduce_ad_spend",
                    "channel": line["channel"],
                    "sku": None,
                    "change_pct": 20,
                    "rationale": f"Advertising takes {line['ad_share_pct']}% of {line['channel']} revenue.",
                    "evidence": {"metric": "ad_share_pct", "value": line["ad_share_pct"]},
                }
            )
        if line["return_rate_pct"] > 15:
            actions.append(
                {
                    "action": "investigate_returns",
                    "channel": line["channel"],
                    "sku": None,
                    "change_pct": None,
                    "rationale": f"{line['return_rate_pct']}% of {line['channel']} orders come back.",
                    "evidence": {"metric": "return_rate_pct", "value": line["return_rate_pct"]},
                }
            )
    for line in sorted(report["skus"], key=lambda item: item["margin_pct"]):
        if line["contribution"] < 0:
            actions.append(
                {
                    "action": "pause_sku_on_channel",
                    "channel": line["channel"],
                    "sku": line["sku"],
                    "change_pct": None,
                    "rationale": f"{line['sku']} loses money on {line['channel']} after fees, shipping and ads.",
                    "evidence": {"metric": "contribution", "value": line["contribution"]},
                }
            )
        elif line["margin_pct"] < 8:
            actions.append(
                {
                    "action": "raise_price",
                    "channel": line["channel"],
                    "sku": line["sku"],
                    "change_pct": 5,
                    "rationale": f"{line['sku']} earns a thin {line['margin_pct']}% margin on {line['channel']}.",
                    "evidence": {"metric": "margin_pct", "value": line["margin_pct"]},
                }
            )
    return json.dumps({"actions": actions[:6]})
