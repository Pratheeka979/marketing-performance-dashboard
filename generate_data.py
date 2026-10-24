from pathlib import Path

import numpy as np
import pandas as pd


# The same seed produces the same sample data each time.
rng = np.random.default_rng(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

dates = pd.date_range("2026-01-01", "2026-06-30", freq="D")

# Example assumptions used only to simulate channel behaviour.
channel_settings = {
    "Google Ads": {
        "ctr": 0.045,
        "conversion_rate": 0.055,
        "cost_per_click": 18,
    },
    "Instagram": {
        "ctr": 0.030,
        "conversion_rate": 0.035,
        "cost_per_click": 12,
    },
    "Facebook": {
        "ctr": 0.025,
        "conversion_rate": 0.030,
        "cost_per_click": 10,
    },
    "Email": {
        "ctr": 0.065,
        "conversion_rate": 0.060,
        "cost_per_click": 3,
    },
}

campaigns = ["Brand Awareness", "Product Launch", "Seasonal Offer"]
rows = []

for date in dates:
    for channel, settings in channel_settings.items():
        for campaign in campaigns:
            impressions = int(rng.integers(1500, 9000))

            click_probability = float(np.clip(
                settings["ctr"] * rng.uniform(0.75, 1.25),
                0,
                1,
            ))
            clicks = int(rng.binomial(impressions, click_probability))

            purchase_probability = float(np.clip(
                settings["conversion_rate"] * rng.uniform(0.70, 1.30),
                0,
                1,
            ))
            conversions = int(rng.binomial(clicks, purchase_probability))

            spend = round(
                clicks
                * settings["cost_per_click"]
                * rng.uniform(0.85, 1.15),
                2,
            )

            revenue = round(
                conversions * rng.uniform(700, 1600),
                2,
            )

            # Engagement actions can include clicks, likes,
            # comments, and shares. These are actions, not unique users.
            engagements = clicks + int(
                rng.binomial(impressions - clicks, 0.015)
            )

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "channel": channel,
                "campaign": campaign,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "engagements": engagements,
                "spend": spend,
                "revenue": revenue,
            })

df = pd.DataFrame(rows)

# Add known data-quality issues so the cleaning stage is demonstrable.
raw = pd.concat([df, df.iloc[:10]], ignore_index=True)
raw.loc[20:24, "channel"] = " Instagram "
raw.loc[30:32, "revenue"] = np.nan

output_path = DATA_DIR / "marketing_raw.csv"
raw.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Raw rows: {len(raw)}")
print("Source: synthetic data generated for an academic project.")