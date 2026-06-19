"""Generate a small CAD-like sample for smoke testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_cad(path: str | Path, seed: int = 7, event_count: int = 250) -> Path:
    rng = np.random.default_rng(seed)
    call_types = np.array(["Traffic Stop", "Burglary", "Welfare Check", "Noise", "Assault"])
    priorities = np.array([1, 2, 3, 4])
    dispositions = np.array(["Report Taken", "Arrest", "Unfounded", "Warning", "No Service"])
    units = np.array([f"CAR-{index:03d}" for index in range(1, 45)])

    rows = []
    start = pd.Timestamp("2025-01-01")
    for event_index in range(1, event_count + 1):
        call_type = rng.choice(call_types, p=[0.34, 0.12, 0.21, 0.22, 0.11])
        priority = int(rng.choice(priorities, p=[0.08, 0.2, 0.45, 0.27]))
        disposition = rng.choice(dispositions, p=[0.28, 0.08, 0.18, 0.32, 0.14])
        day_offset = int(rng.integers(0, 45))
        hour = int(rng.choice(np.arange(24), p=_hour_distribution()))
        minute = int(rng.integers(0, 60))
        received = start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)
        unit_count = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
        location = f"{int(rng.integers(100, 9999))} Block Main St"
        latitude = 39.76 + rng.normal(0, 0.04)
        longitude = -86.15 + rng.normal(0, 0.04)

        for unit in rng.choice(units, size=unit_count, replace=False):
            dispatch = received + pd.Timedelta(minutes=max(0, rng.normal(2, 1)))
            arrival = dispatch + pd.Timedelta(minutes=max(1, rng.normal(8 + priority, 3)))
            clear = arrival + pd.Timedelta(minutes=max(5, rng.normal(42, 18)))
            rows.append(
                {
                    "EventNumber": f"E{event_index:06d}",
                    "ReceivedDateTime": received,
                    "IncidentAddress": location,
                    "Latitude": round(float(latitude), 6),
                    "Longitude": round(float(longitude), 6),
                    "CallType": call_type,
                    "Priority": priority,
                    "Disposition": disposition,
                    "Unit": unit,
                    "DispatchTime": dispatch,
                    "ArrivalTime": arrival,
                    "ClearTime": clear,
                }
            )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def _hour_distribution() -> np.ndarray:
    hours = np.arange(24)
    afternoon_peak = np.exp(-((hours - 16) ** 2) / 25)
    late_peak = 0.45 * np.exp(-((hours - 22) ** 2) / 14)
    baseline = np.full(24, 0.3)
    weights = baseline + afternoon_peak + late_peak
    return weights / weights.sum()
