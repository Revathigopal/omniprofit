"""Generate the synthetic sample data in ``data/``. Seeded, so the output is reproducible."""

import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
random.seed(7)

SKUS = {"KURTA-01": (899, 380), "SAREE-02": (2499, 1250), "DUPATTA-03": (349, 210), "JEWEL-04": (1299, 420)}
CHANNELS = {
    "amazon": {"commission_pct": 17.0, "fixed_fee": 25, "shipping_per_unit": 65, "payment_fee_pct": 0.0, "return_handling": 120},
    "flipkart": {"commission_pct": 15.0, "fixed_fee": 20, "shipping_per_unit": 60, "payment_fee_pct": 0.0, "return_handling": 110},
    "website": {"commission_pct": 0.0, "fixed_fee": 0, "shipping_per_unit": 55, "payment_fee_pct": 2.0, "return_handling": 90},
    "ondc": {"commission_pct": 5.0, "fixed_fee": 5, "shipping_per_unit": 58, "payment_fee_pct": 1.0, "return_handling": 90},
}
WEIGHTS = {"amazon": 0.4, "flipkart": 0.3, "website": 0.2, "ondc": 0.1}
RETURN_RATE = {"amazon": 0.14, "flipkart": 0.18, "website": 0.06, "ondc": 0.08}
AD_SPEND = {"amazon": 21000, "flipkart": 24000, "website": 9000, "ondc": 1500}

DATA.mkdir(exist_ok=True)
with (DATA / "orders.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["order_id", "date", "channel", "sku", "units", "unit_price", "returned"])
    for number in range(1, 601):
        channel = random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
        sku = random.choice(list(SKUS))
        price = SKUS[sku][0] * random.choice([1.0, 1.0, 0.9, 0.8])
        day = date(2026, 4, 1) + timedelta(days=random.randrange(30))
        returned = int(random.random() < RETURN_RATE[channel])
        writer.writerow([f"ORD{number:04d}", day.isoformat(), channel, sku, random.choice([1, 1, 1, 2]), round(price), returned])

with (DATA / "sku_costs.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["sku", "cogs"])
    for sku, (_, cogs) in SKUS.items():
        writer.writerow([sku, cogs])

(DATA / "channel_fees.json").write_text(json.dumps(CHANNELS, indent=2) + "\n")
(DATA / "ad_spend.json").write_text(json.dumps(AD_SPEND, indent=2) + "\n")
print("wrote sample data to", DATA)
