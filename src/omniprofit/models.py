"""Data contracts shared by the agents."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Order(Strict):
    order_id: str = Field(min_length=1)
    date: date
    channel: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    units: int = Field(gt=0)
    unit_price: float = Field(gt=0)
    returned: bool


class ChannelFees(Strict):
    commission_pct: float = Field(ge=0, le=100)
    fixed_fee: float = Field(ge=0)
    shipping_per_unit: float = Field(ge=0)
    payment_fee_pct: float = Field(ge=0, le=100)
    return_handling: float = Field(ge=0)


class Dataset(Strict):
    orders: list[Order]
    fees: dict[str, ChannelFees]
    cogs: dict[str, float]
    ad_spend: dict[str, float]
    rejected_rows: list[str] = []


class OrderEconomics(Strict):
    order_id: str
    channel: str
    sku: str
    units: int
    returned: bool
    revenue: float
    cogs: float
    channel_fees: float
    shipping: float
    return_cost: float


class ProfitLine(Strict):
    """Profitability of one channel, or of one SKU within a channel."""

    channel: str
    sku: str | None = None
    orders: int
    units: int
    revenue: float
    cogs: float
    channel_fees: float
    shipping: float
    return_cost: float
    ad_spend: float
    contribution: float
    margin_pct: float
    return_rate_pct: float
    ad_share_pct: float


class ProfitReport(Strict):
    channels: list[ProfitLine]
    skus: list[ProfitLine]
    total_revenue: float
    total_contribution: float


Metric = Literal["margin_pct", "contribution", "return_rate_pct", "ad_share_pct"]


class Evidence(Strict):
    metric: Metric
    value: float


class Action(Strict):
    action: Literal["raise_price", "reduce_ad_spend", "pause_sku_on_channel", "investigate_returns"]
    channel: str
    sku: str | None = None
    change_pct: float | None = Field(default=None, description="Size of the change, in percent, where it applies.")
    rationale: str = Field(min_length=10)
    evidence: Evidence


class Advice(Strict):
    actions: list[Action]


class RejectedAction(Strict):
    action: Action
    reasons: list[str]


class GuardedAdvice(Strict):
    approved: list[Action]
    rejected: list[RejectedAction]
