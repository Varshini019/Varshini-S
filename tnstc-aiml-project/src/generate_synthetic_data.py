from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_routes(num_routes: int, rng: random.Random) -> pd.DataFrame:
    route_types = ["urban", "suburban", "rural"]
    routes: List[dict] = []
    for r in range(1, num_routes + 1):
        route_id = f"R{r:02d}"
        route_type = rng.choices(route_types, weights=[0.6, 0.3, 0.1], k=1)[0]
        length_km = round(rng.uniform(8.0, 45.0), 1)
        routes.append({
            "route_id": route_id,
            "route_type": route_type,
            "length_km": length_km,
        })
    return pd.DataFrame(routes)


def generate_stops(routes_df: pd.DataFrame, stops_per_route: int, rng: random.Random) -> pd.DataFrame:
    # Center around Chennai (approx), small random offsets
    base_lat, base_lon = 13.0827, 80.2707
    stops: List[dict] = []
    for _, row in routes_df.iterrows():
        route_id = row["route_id"]
        for s in range(1, stops_per_route + 1):
            stop_id = f"{route_id}_S{s:02d}"
            lat = base_lat + rng.uniform(-0.25, 0.25)
            lon = base_lon + rng.uniform(-0.25, 0.25)
            stops.append({
                "route_id": route_id,
                "stop_id": stop_id,
                "stop_index": s,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
            })
    return pd.DataFrame(stops)


def generate_calendar(start_date: datetime, num_days: int, rng: random.Random) -> pd.DataFrame:
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    # Randomly choose some holiday dates
    num_holidays = max(3, num_days // 30)
    holiday_dates = set(rng.sample(dates, k=min(num_holidays, len(dates))))

    calendar_rows = []
    for d in dates:
        day_of_week = d.weekday()  # 0=Mon, 6=Sun
        is_holiday = d in holiday_dates
        # Simple synthetic weather index [0,1]
        weather_index = max(0.0, min(1.0, rng.gauss(0.5, 0.2)))
        calendar_rows.append({
            "date": d.date().isoformat(),
            "day_of_week": day_of_week,
            "is_holiday": int(is_holiday),
            "month": d.month,
            "weather_index": round(weather_index, 3),
        })
    return pd.DataFrame(calendar_rows)


def _time_of_day_multiplier(hour: int) -> float:
    # Peak hours morning/evening, low at late night
    if 7 <= hour <= 10:
        return 1.6
    if 17 <= hour <= 20:
        return 1.5
    if 22 <= hour <= 23 or 0 <= hour <= 5:
        return 0.2
    if 11 <= hour <= 15:
        return 0.85
    return 1.0


def generate_ridership(
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    rows: List[dict] = []

    # Base demand per route type
    base_by_type = {
        "urban": (120, 220),
        "suburban": (60, 140),
        "rural": (20, 70),
    }

    calendar_df_local = calendar_df.copy()
    calendar_df_local["date"] = pd.to_datetime(calendar_df_local["date"]).dt.date

    for _, route in routes_df.iterrows():
        route_id = route["route_id"]
        route_type = route["route_type"]
        base_low, base_high = base_by_type[route_type]
        base_route = rng.uniform(base_low, base_high)

        stops_for_route = stops_df[stops_df["route_id"] == route_id]

        for _, stop in stops_for_route.iterrows():
            stop_id = stop["stop_id"]
            stop_index = stop["stop_index"]
            # More demand near middle of route, taper ends
            stop_position_weight = 1.0 - abs((stop_index - (stops_for_route.shape[0] / 2)) / (stops_for_route.shape[0] / 2)) * 0.4

            for _, cal in calendar_df_local.iterrows():
                date_obj = cal["date"]
                day_of_week = cal["day_of_week"]
                is_holiday = cal["is_holiday"]
                weather_index = cal["weather_index"]

                # Weekday/weekend effects
                weekday_multiplier = 1.15 if day_of_week < 5 else 0.85
                holiday_multiplier = 0.7 if is_holiday == 1 else 1.0
                weather_multiplier = 0.9 + 0.3 * (1.0 - abs(weather_index - 0.5) * 2)  # better weather -> slightly higher demand

                for hour in range(5, 24):
                    tod_mult = _time_of_day_multiplier(hour)
                    mean_demand = base_route * stop_position_weight * weekday_multiplier * holiday_multiplier * weather_multiplier * tod_mult

                    # Add mild trend/seasonality noise
                    seasonal = 1.0 + 0.1 * math.sin((hour / 24.0) * 2 * math.pi)
                    mean_demand *= seasonal

                    # Sample from Poisson with safety floor
                    lam = max(1.0, mean_demand)
                    passengers = int(np.random.poisson(lam))

                    rows.append({
                        "date": pd.to_datetime(date_obj).date().isoformat(),
                        "hour": hour,
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "passengers_count": passengers,
                    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TNSTC-like ridership data")
    parser.add_argument("--num_routes", type=int, default=12)
    parser.add_argument("--stops_per_route", type=int, default=15)
    parser.add_argument("--num_days", type=int, default=90)
    parser.add_argument("--start_date", type=str, default="2025-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    routes_df = generate_routes(args.num_routes, rng)
    stops_df = generate_stops(routes_df, args.stops_per_route, rng)
    calendar_df = generate_calendar(datetime.fromisoformat(args.start_date), args.num_days, rng)

    ridership_df = generate_ridership(routes_df, stops_df, calendar_df, rng)

    # Save files
    routes_df.to_csv(data_dir / "routes.csv", index=False)
    stops_df.to_csv(data_dir / "stops.csv", index=False)
    calendar_df.to_csv(data_dir / "calendar.csv", index=False)
    ridership_df.to_csv(data_dir / "ridership.csv", index=False)

    print(f"Saved: {data_dir}/routes.csv, stops.csv, calendar.csv, ridership.csv")


if __name__ == "__main__":
    main()