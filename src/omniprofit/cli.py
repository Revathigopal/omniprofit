"""Command-line interface: ``omniprofit report``, ``recommend`` and ``serve-mcp``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm import LLMError, get_llm
from .mock import mock_handler
from .models import Action, ProfitLine
from .orchestrator import Orchestrator


def table(lines: list[ProfitLine]) -> str:
    header = f"{'channel':<10}{'sku':<12}{'orders':>7}{'revenue':>11}{'contrib.':>11}{'margin%':>9}{'returns%':>10}{'ads%':>7}"
    rows = [header, "-" * len(header)]
    for line in lines:
        rows.append(
            f"{line.channel:<10}{(line.sku or 'all'):<12}{line.orders:>7}{line.revenue:>11,.0f}"
            f"{line.contribution:>11,.0f}{line.margin_pct:>9.1f}{line.return_rate_pct:>10.1f}{line.ad_share_pct:>7.1f}"
        )
    return "\n".join(rows)


def describe(action: Action) -> str:
    target = action.channel if action.sku is None else f"{action.sku} on {action.channel}"
    size = f" by {action.change_pct:g}%" if action.change_pct else ""
    return f"{action.action}{size}: {target}. {action.rationale}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omniprofit", description="Profit intelligence across sales channels.")
    parser.add_argument("command", choices=["report", "recommend", "serve-mcp"])
    parser.add_argument("--data", type=Path, default=Path("data"), help="Folder with orders.csv and cost files.")
    parser.add_argument("--provider", help="mock (default), ollama, groq, openrouter or openai.")
    parser.add_argument("--model")
    parser.add_argument("--skus", action="store_true", help="Also print the SKU-level table.")
    args = parser.parse_args(argv)

    if args.command == "serve-mcp":
        from .mcp_server import build_server

        build_server(str(args.data)).run()
        return 0

    try:
        llm = get_llm(args.provider, args.model, mock_handler=mock_handler) if args.command == "recommend" else None
        result = Orchestrator(llm).run(args.data, advise=args.command == "recommend")
    except (LLMError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{result.orders_loaded} orders loaded, {len(result.rejected_rows)} rejected")
    print(table(result.report.channels))
    print(f"\nTotal revenue {result.report.total_revenue:,.0f}, contribution {result.report.total_contribution:,.0f}")
    if args.skus:
        print("\n" + table(result.report.skus))
    if result.advice is not None:
        print("\nApproved actions")
        for action in result.advice.approved:
            print("  - " + describe(action))
        if result.advice.rejected:
            print("\nRejected by guardrails")
            for item in result.advice.rejected:
                print("  - " + describe(item.action))
                for reason in item.reasons:
                    print("      reason: " + reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
