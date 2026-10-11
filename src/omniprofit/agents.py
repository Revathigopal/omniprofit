"""Four agents with one job each: ingest, cost, profit, advise.

The first three are deterministic. Only the advisor uses a language model,
and everything it proposes passes through ``guardrails.review`` before it is
shown to anyone.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .llm import LLM, extract_json
from .models import (
    Advice,
    ChannelFees,
    Dataset,
    Order,
    OrderEconomics,
    ProfitLine,
    ProfitReport,
)


class IngestAgent:
    """Loads and validates raw files. Bad rows are set aside and reported, never guessed at."""

    def run(self, data_dir: Path) -> Dataset:
        data_dir = Path(data_dir)
        fees = {
            name: ChannelFees.model_validate(spec)
            for name, spec in json.loads((data_dir / "channel_fees.json").read_text()).items()
        }
        with (data_dir / "sku_costs.csv").open(newline="") as handle:
            cogs = {row["sku"]: float(row["cogs"]) for row in csv.DictReader(handle)}
        ad_spend = {k: float(v) for k, v in json.loads((data_dir / "ad_spend.json").read_text()).items()}

        orders, rejected = [], []
        with (data_dir / "orders.csv").open(newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    order = Order.model_validate({**row, "returned": row.get("returned") in {"1", "true", "True"}})
                except ValidationError as exc:
                    rejected.append(f"line {line_number}: {exc.errors()[0]['loc'][0]} {exc.errors()[0]['msg']}")
                    continue
                if order.channel not in fees:
                    rejected.append(f"line {line_number}: unknown channel '{order.channel}'")
                elif order.sku not in cogs:
                    rejected.append(f"line {line_number}: no cost on file for SKU '{order.sku}'")
                else:
                    orders.append(order)
        return Dataset(orders=orders, fees=fees, cogs=cogs, ad_spend=ad_spend, rejected_rows=rejected)


class CostAgent:
    """Works out what each order earned and cost.

    A returned order earns no revenue and recovers its goods, but still pays
    outbound shipping and a return-handling charge. Marketplace commission is
    assumed to be refunded on returns; the fixed fee is not.
    """

    def run(self, dataset: Dataset) -> list[OrderEconomics]:
        lines = []
        for order in dataset.orders:
            fees = dataset.fees[order.channel]
            gross = order.units * order.unit_price
            shipping = order.units * fees.shipping_per_unit
            if order.returned:
                revenue, cogs = 0.0, 0.0
                channel_fees = fees.fixed_fee
                return_cost = fees.return_handling
            else:
                revenue, cogs = gross, order.units * dataset.cogs[order.sku]
                channel_fees = gross * (fees.commission_pct + fees.payment_fee_pct) / 100 + fees.fixed_fee
                return_cost = 0.0
            lines.append(
                OrderEconomics(
                    order_id=order.order_id,
                    channel=order.channel,
                    sku=order.sku,
                    units=order.units,
                    returned=order.returned,
                    revenue=revenue,
                    cogs=cogs,
                    channel_fees=channel_fees,
                    shipping=shipping,
                    return_cost=return_cost,
                )
            )
        return lines


def _line(channel: str, sku: str | None, items: list[OrderEconomics], ad_spend: float) -> ProfitLine:
    revenue = sum(i.revenue for i in items)
    cogs = sum(i.cogs for i in items)
    channel_fees = sum(i.channel_fees for i in items)
    shipping = sum(i.shipping for i in items)
    return_cost = sum(i.return_cost for i in items)
    contribution = revenue - cogs - channel_fees - shipping - return_cost - ad_spend
    return ProfitLine(
        channel=channel,
        sku=sku,
        orders=len(items),
        units=sum(i.units for i in items),
        revenue=round(revenue, 2),
        cogs=round(cogs, 2),
        channel_fees=round(channel_fees, 2),
        shipping=round(shipping, 2),
        return_cost=round(return_cost, 2),
        ad_spend=round(ad_spend, 2),
        contribution=round(contribution, 2),
        margin_pct=round(100 * contribution / revenue, 2) if revenue else 0.0,
        return_rate_pct=round(100 * sum(i.returned for i in items) / len(items), 2) if items else 0.0,
        ad_share_pct=round(100 * ad_spend / revenue, 2) if revenue else 0.0,
    )


class ProfitAgent:
    """Rolls order economics up to channel and SKU-within-channel contribution margin.

    Channel ad spend is allocated to SKUs in proportion to their revenue.
    """

    def run(self, economics: list[OrderEconomics], ad_spend: dict[str, float]) -> ProfitReport:
        by_channel: dict[str, list[OrderEconomics]] = defaultdict(list)
        by_sku: dict[tuple[str, str], list[OrderEconomics]] = defaultdict(list)
        for item in economics:
            by_channel[item.channel].append(item)
            by_sku[(item.channel, item.sku)].append(item)

        channels = [_line(name, None, items, ad_spend.get(name, 0.0)) for name, items in sorted(by_channel.items())]
        channel_revenue = {line.channel: line.revenue for line in channels}
        skus = []
        for (channel, sku), items in sorted(by_sku.items()):
            revenue = sum(i.revenue for i in items)
            share = revenue / channel_revenue[channel] if channel_revenue[channel] else 0.0
            skus.append(_line(channel, sku, items, ad_spend.get(channel, 0.0) * share))
        return ProfitReport(
            channels=channels,
            skus=skus,
            total_revenue=round(sum(line.revenue for line in channels), 2),
            total_contribution=round(sum(line.contribution for line in channels), 2),
        )


ADVISOR_PROMPT = """AGENT:advisor
You advise an online seller on profit. You are given contribution-margin figures per channel and per SKU within each channel.
Propose at most 6 actions. Every action must cite one figure, copied exactly from the data, as its evidence.
Allowed actions: raise_price, reduce_ad_spend, pause_sku_on_channel, investigate_returns.
For raise_price and reduce_ad_spend give change_pct. Use sku=null for channel-level actions.
Reply with one JSON object only, matching this schema:
{schema}"""


class AdvisorAgent:
    """Asks a language model for actions. Its output is untrusted until reviewed."""

    def __init__(self, llm: LLM, max_attempts: int = 2) -> None:
        self.llm = llm
        self.max_attempts = max_attempts

    def run(self, report: ProfitReport) -> Advice:
        messages = [
            {"role": "system", "content": ADVISOR_PROMPT.format(schema=json.dumps(Advice.model_json_schema()))},
            {"role": "user", "content": report.model_dump_json()},
        ]
        error = ""
        for _ in range(self.max_attempts):
            reply = self.llm.complete(messages)
            try:
                return Advice.model_validate(extract_json(reply))
            except (ValueError, ValidationError) as exc:
                error = str(exc)
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"Rejected: {error}\nSend corrected JSON."},
                ]
        raise RuntimeError(f"advisor did not return valid advice: {error}")
