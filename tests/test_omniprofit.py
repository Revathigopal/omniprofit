import asyncio
import json
from pathlib import Path

import pytest

from omniprofit.agents import AdvisorAgent, CostAgent, IngestAgent, ProfitAgent
from omniprofit.guardrails import Limits, review
from omniprofit.llm import MockLLM
from omniprofit.mcp_server import build_server, channel_profit, recommend, sku_profit
from omniprofit.mock import mock_handler
from omniprofit.models import Advice
from omniprofit.orchestrator import Orchestrator

SAMPLE = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def tiny(tmp_path):
    (tmp_path / "orders.csv").write_text(
        "order_id,date,channel,sku,units,unit_price,returned\n"
        "A1,2026-04-01,market,SHIRT,2,500,0\n"
        "A2,2026-04-02,market,SHIRT,1,500,1\n"
        "A3,2026-04-02,site,SHIRT,1,500,0\n"
        "A4,2026-04-03,market,SHIRT,0,500,0\n"  # zero units: invalid
        "A5,2026-04-03,bazaar,SHIRT,1,500,0\n"  # unknown channel
        "A6,2026-04-03,site,HAT,1,200,0\n"  # no cost on file
    )
    (tmp_path / "sku_costs.csv").write_text("sku,cogs\nSHIRT,200\n")
    (tmp_path / "channel_fees.json").write_text(
        json.dumps(
            {
                "market": {"commission_pct": 10, "fixed_fee": 20, "shipping_per_unit": 50, "payment_fee_pct": 0, "return_handling": 100},
                "site": {"commission_pct": 0, "fixed_fee": 0, "shipping_per_unit": 40, "payment_fee_pct": 2, "return_handling": 80},
            }
        )
    )
    (tmp_path / "ad_spend.json").write_text(json.dumps({"market": 100, "site": 50}))
    return tmp_path


def report_for(folder):
    dataset = IngestAgent().run(folder)
    return dataset, ProfitAgent().run(CostAgent().run(dataset), dataset.ad_spend)


def test_ingest_sets_aside_bad_rows_with_reasons(tiny):
    dataset, _ = report_for(tiny)
    assert [o.order_id for o in dataset.orders] == ["A1", "A2", "A3"]
    assert len(dataset.rejected_rows) == 3
    assert "unknown channel 'bazaar'" in dataset.rejected_rows[1]
    assert "no cost on file for SKU 'HAT'" in dataset.rejected_rows[2]


def test_contribution_margin_matches_hand_calculation(tiny):
    _, report = report_for(tiny)
    market = next(c for c in report.channels if c.channel == "market")
    # A1: revenue 1000, cogs 400, fees 10% + 20 = 120, shipping 100
    # A2 (returned): revenue 0, cogs 0, fixed fee 20, shipping 50, return handling 100
    assert market.revenue == 1000
    assert market.cogs == 400
    assert market.channel_fees == 140
    assert market.shipping == 150
    assert market.return_cost == 100
    assert market.contribution == 1000 - 400 - 140 - 150 - 100 - 100  # last 100 is ad spend
    assert market.margin_pct == 11.0
    assert market.return_rate_pct == 50.0
    site = next(c for c in report.channels if c.channel == "site")
    assert site.contribution == 500 - 200 - 10 - 40 - 50
    assert report.total_contribution == market.contribution + site.contribution


def test_sku_ad_spend_allocation_adds_up_to_channel_spend():
    _, report = report_for(SAMPLE)
    for channel in report.channels:
        allocated = sum(line.ad_spend for line in report.skus if line.channel == channel.channel)
        assert allocated == pytest.approx(channel.ad_spend, abs=0.05)
        total = sum(line.contribution for line in report.skus if line.channel == channel.channel)
        assert total == pytest.approx(channel.contribution, abs=0.05)


def advice(*actions):
    return Advice.model_validate({"actions": list(actions)})


def base(**overrides):
    action = {
        "action": "raise_price",
        "channel": "market",
        "sku": None,
        "change_pct": 5,
        "rationale": "Margin on this channel is thin.",
        "evidence": {"metric": "margin_pct", "value": 11.0},
    }
    return {**action, **overrides}


def test_guardrails_approve_grounded_action(tiny):
    _, report = report_for(tiny)
    result = review(advice(base()), report)
    assert len(result.approved) == 1 and not result.rejected


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"evidence": {"metric": "margin_pct", "value": 35.0}}, "cited margin_pct=35 but the data says 11"),
        ({"change_pct": 40}, "exceeds the 10% limit for raise_price"),
        ({"channel": "tiktok"}, "does not exist in the data"),
        ({"action": "pause_sku_on_channel", "change_pct": None}, "whole channels cannot be paused"),
        (
            {"action": "pause_sku_on_channel", "sku": "SHIRT", "change_pct": None},
            "pausing would lose money",
        ),
    ],
)
def test_guardrails_reject_with_reason(tiny, overrides, expected):
    _, report = report_for(tiny)
    result = review(advice(base(**overrides)), report)
    assert not result.approved
    assert any(expected in reason for reason in result.rejected[0].reasons)


def test_guardrails_block_pausing_a_major_revenue_line(tiny):
    _, report = report_for(tiny)
    line = next(s for s in report.skus if s.channel == "market")
    action = base(action="pause_sku_on_channel", sku="SHIRT", change_pct=None, evidence={"metric": "contribution", "value": line.contribution})
    result = review(advice(action), report, Limits(pause_requires_negative_contribution=False))
    assert "above the 15% pause limit" in result.rejected[0].reasons[0]


def test_unknown_action_type_never_reaches_the_guardrails(tiny):
    _, report = report_for(tiny)
    llm = MockLLM(lambda messages: json.dumps({"actions": [base(action="delete_all_listings")]}))
    with pytest.raises(RuntimeError, match="did not return valid advice"):
        AdvisorAgent(llm).run(report)


def test_end_to_end_on_sample_data():
    result = Orchestrator(MockLLM(mock_handler)).run(SAMPLE)
    assert result.orders_loaded == 600
    assert result.advice.approved and not result.advice.rejected
    paused = {(a.sku, a.channel) for a in result.advice.approved if a.action == "pause_sku_on_channel"}
    losing = {(s.sku, s.channel) for s in result.report.skus if s.contribution < 0}
    assert paused == losing


def test_mcp_tool_functions():
    assert {line["channel"] for line in channel_profit(str(SAMPLE))} == {"amazon", "flipkart", "ondc", "website"}
    assert len(sku_profit(str(SAMPLE), "ondc")) == 4
    with pytest.raises(ValueError):
        sku_profit(str(SAMPLE), "nowhere")
    assert "approved" in recommend(str(SAMPLE))


def test_mcp_server_registers_three_tools():
    pytest.importorskip("mcp")
    server = build_server(str(SAMPLE))
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"get_channel_profit", "get_sku_profit", "get_recommendations"}
