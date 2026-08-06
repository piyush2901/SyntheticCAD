from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

from syntheticcad.sensitive import (
    add_synthetic_identifiers,
    assess_dataframe,
    bucket_rare_categories,
    direct_identifier_overlap,
    protect_synthetic_identifier_values,
)
from syntheticcad.tabular import (
    _distance_privacy_screens,
    _repair_learned_constraints,
    estimate_runtime,
)
from syntheticcad.tabular_dashboard import write_tabular_dashboard


class SensitivePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = pd.DataFrame(
            {
                "first_name": ["Alex", "Blair", "Casey", "Drew"],
                "date_of_birth": [
                    "1980-01-01",
                    "1990-02-02",
                    "1975-03-03",
                    "2000-04-04",
                ],
                "age_at_offense": [45, 35, 50, 25],
                "race": ["A", "A", "B", "C"],
                "offense": ["Theft", "Theft", "Fraud", "Assault"],
                "offense_date": [
                    "2025-01-02",
                    "2025-02-03",
                    "2025-03-04",
                    "2025-04-05",
                ],
            }
        )

    def test_field_roles_prioritize_quasi_identifiers(self) -> None:
        roles = {
            item.column: item.role for item in assess_dataframe(self.source)
        }
        self.assertEqual(roles["first_name"], "direct_identifier")
        self.assertEqual(roles["date_of_birth"], "direct_identifier")
        self.assertEqual(roles["age_at_offense"], "quasi_identifier")
        self.assertEqual(roles["offense"], "sensitive_attribute")
        self.assertEqual(roles["offense_date"], "quasi_identifier")

    def test_identifier_replacements_have_no_source_overlap(self) -> None:
        roles = {
            item.column: item.role for item in assess_dataframe(self.source)
        }
        modeled = self.source[["age_at_offense", "race", "offense", "offense_date"]]
        synthetic = add_synthetic_identifiers(
            modeled,
            list(self.source.columns),
            roles,
            seed=7,
        )
        synthetic = protect_synthetic_identifier_values(
            self.source,
            synthetic,
            roles,
            seed=7,
        )
        overlap = direct_identifier_overlap(self.source, synthetic, roles)
        self.assertEqual(
            overlap["exact_identity_combination"]["matching_synthetic_rows"],
            0,
        )
        for field in overlap["fields"].values():
            self.assertEqual(field["synthetic_values_matching_source"], 0)

    def test_rare_bucketing_does_not_modify_datetime_columns(self) -> None:
        frame = self.source.copy()
        frame["offense_date"] = pd.to_datetime(frame["offense_date"])
        roles = {
            item.column: item.role for item in assess_dataframe(self.source)
        }
        bucketed, _ = bucket_rare_categories(frame, roles, threshold=2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(bucketed["offense_date"]))

    def test_detected_constraints_are_repaired(self) -> None:
        real = pd.DataFrame(
            {
                "admission_date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                "discharge_date": pd.to_datetime(["2025-01-03", "2025-01-04"]),
                "length_of_stay_days": [2, 1],
                "diagnosis_code": ["A", "B"],
                "diagnosis_description": ["Alpha", "Beta"],
            }
        )
        synthetic = pd.DataFrame(
            {
                "admission_date": pd.to_datetime(["2025-02-01", "2025-02-03"]),
                "discharge_date": pd.to_datetime(["2025-02-20", "2025-02-20"]),
                "length_of_stay_days": [2, 1],
                "diagnosis_code": ["A", "B"],
                "diagnosis_description": ["Beta", "Alpha"],
            }
        )
        repaired, details = _repair_learned_constraints(
            real,
            synthetic,
            {
                "admission_date": "datetime",
                "discharge_date": "datetime",
                "length_of_stay_days": "numerical",
                "diagnosis_code": "categorical",
                "diagnosis_description": "categorical",
            },
        )
        expected = repaired["admission_date"] + pd.to_timedelta(
            repaired["length_of_stay_days"],
            unit="D",
        )
        self.assertTrue((repaired["discharge_date"] == expected).all())
        self.assertEqual(repaired["diagnosis_description"].tolist(), ["Alpha", "Beta"])
        self.assertTrue(any(item["type"] == "functional_dependency" for item in details))

    def test_runtime_estimate_increases_with_rows_and_epochs(self) -> None:
        small = estimate_runtime(2_000, 5, "gaussian_copula")
        large = estimate_runtime(20_000, 5, "gaussian_copula")
        ctgan = estimate_runtime(20_000, 5, "ctgan", epochs=100)
        self.assertLess(
            small["estimated_seconds_high"],
            large["estimated_seconds_high"],
        )
        self.assertLess(
            large["estimated_seconds_high"],
            ctgan["estimated_seconds_high"],
        )

    def test_distance_screens_disclose_source_reference_scope(self) -> None:
        real = pd.DataFrame(
            {
                "age": list(range(100)),
                "group": ["A" if index % 2 else "B" for index in range(100)],
            }
        )
        synthetic = pd.DataFrame(
            {
                "age": [value + 0.25 for value in range(100)],
                "group": ["A" if index % 2 else "B" for index in range(100)],
            }
        )
        screens = _distance_privacy_screens(
            real,
            synthetic,
            {"age": "numerical", "group": "categorical"},
            seed=42,
            sample_size=25,
        )
        self.assertTrue(screens["available"])
        self.assertIsNotNone(
            screens["distance_to_closest_record"]["median_distance_ratio"]
        )
        self.assertIsNotNone(
            screens["nearest_neighbor_distance_ratio"]["synthetic_to_real_train"][
                "median"
            ]
        )
        self.assertIsNotNone(
            screens["distance_to_closest_record"][
                "source_reference_to_synthetic"
            ]["p01"]
        )
        self.assertIn("not an independent holdout", screens["comparison_data"])

    def test_dashboard_omits_identifier_values(self) -> None:
        real = self.source[
            ["age_at_offense", "race", "offense", "offense_date"]
        ].copy()
        synthetic = real.copy()
        report = {
            "pipeline": {
                "source_rows": 4,
                "synthetic_rows": 4,
                "modeled_columns": list(real.columns),
                "excluded_identifier_columns": ["first_name", "date_of_birth"],
                "method": "gaussian_copula",
                "seed": 42,
                "rare_category_threshold": 5,
            },
            "runtime": {
                "fit_seconds": 1,
                "sample_seconds": 1,
                "evaluation_seconds": 1,
                "total_seconds": 3,
            },
            "field_assessments": [
                item.to_dict() for item in assess_dataframe(real)
            ],
            "quality": {
                "overall_score": 0.9,
                "diagnostic_score": 1.0,
                "column_shapes": [],
                "column_metrics": [],
            },
            "privacy": {
                "direct_identifier_overlap": {
                    "exact_identity_combination": {
                        "matching_synthetic_rows": 0
                    }
                },
                "exact_modeled_row_overlap": {"matching_synthetic_rows": 0},
                "rare_combination_exposure": {
                    "source_rare_combinations_present_in_synthetic": 0,
                    "presence_rate": 0,
                },
            },
            "claims": {"supported": [], "not_claimed": []},
            "methodology": "Local test.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_tabular_dashboard(
                real,
                synthetic,
                report,
                Path(directory) / "dashboard.html",
            )
            html = path.read_text(encoding="utf-8")
        self.assertIn("Basic Overview", html)
        self.assertIn("Advanced Evidence", html)
        self.assertIn("no real source records", html)
        self.assertNotIn("Real records", html)
        self.assertNotIn('"real_rows"', html)
        self.assertNotIn("Alex", html)


if __name__ == "__main__":
    unittest.main()
