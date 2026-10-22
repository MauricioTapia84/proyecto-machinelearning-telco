import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import deduplicate_customer_profiles
from src.decision import calculate_policy_economics, select_operating_threshold
from src.inference import load_model_artifact, score_customers
from src.group_metrics import calculate_group_metrics
from src.modeling import (
    generate_oof_scores,
    save_model_artifact,
    select_reduced_features,
)
from src.pipeline import (
    PCAKMeansFeatures,
    build_evaluation_pipeline,
    build_pipeline,
    prepare_dataset,
)
from src.sensitivity import compare_deduplication_sensitivity


class DeduplicateCustomerProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            [
                {
                    "customerID": "A",
                    "gender": " Female ",
                    "tenure": "0",
                    "TotalCharges": " ",
                    "Churn": "Yes",
                },
                {
                    "customerID": "B",
                    "gender": "Female",
                    "tenure": "0",
                    "TotalCharges": "0.0",
                    "Churn": "Yes",
                },
                {
                    "customerID": "C",
                    "gender": "Female",
                    "tenure": "0",
                    "TotalCharges": "0.0",
                    "Churn": "No",
                },
            ]
        )

    def test_normalizes_before_removing_profiles_and_keeps_target_in_profile(self) -> None:
        result = deduplicate_customer_profiles(
            self.data,
            id_column="customerID",
            total_charges_column="TotalCharges",
            tenure_column="tenure",
        )

        self.assertEqual(result["customerID"].tolist(), ["A", "C"])
        self.assertEqual(result["TotalCharges"].tolist(), [0.0, 0.0])
        self.assertEqual(result["Churn"].tolist(), ["Yes", "No"])

    def test_prepare_dataset_uses_the_same_profile_policy(self) -> None:
        result = prepare_dataset(self.data)

        self.assertEqual(len(result), 2)
        self.assertNotIn("customerID", result.columns)
        self.assertEqual(result["TotalCharges"].tolist(), [0.0, 0.0])

    def test_pca_kmeans_features_fit_and_transform(self) -> None:
        values = np.random.default_rng(42).normal(size=(40, 6))
        transformer = PCAKMeansFeatures(n_clusters=2)

        transformed = transformer.fit_transform(values)

        self.assertEqual(transformed.shape[0], 40)
        self.assertEqual(transformed.shape[1], transformer.pca_.n_components_ + 2)
        self.assertTrue(np.isfinite(transformed).all())

    def test_reduced_feature_selection_drops_correlated_columns(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(size=60)
        features = pd.DataFrame({
            'signal': signal,
            'signal_duplicate': signal,
            'noise_a': rng.normal(size=60),
            'noise_b': rng.normal(size=60),
        })
        labels = pd.Series((signal > 0).astype(int))

        result = select_reduced_features(
            features,
            labels,
            correlation_threshold=0.95,
            n_features_to_select=2,
        )

        self.assertIn('signal_duplicate', result.dropped_correlated)
        self.assertEqual(len(result.selected_features), 2)
        self.assertNotIn('signal_duplicate', result.selected_features)

    def test_evaluation_pipeline_fits_preprocessing_inside_pipeline(self) -> None:
        rows = []
        for index in range(40):
            churn = index % 2
            rows.append({
                "gender": "Female" if index % 3 else "Male",
                "tenure": index,
                "TotalCharges": str(index * 10),
                "MonthlyCharges": float(index),
                "Churn": churn,
            })
        data = pd.DataFrame(rows)
        X = data.drop(columns="Churn")
        y = data["Churn"]

        pipeline = build_evaluation_pipeline(
            X,
            LogisticRegression(max_iter=1000),
            representation="pca_kmeans",
            use_smote=False,
        )
        pipeline.fit(X.iloc[:30], y.iloc[:30])
        predictions = pipeline.predict(X.iloc[30:])

        self.assertEqual(len(predictions), 10)
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_default_pipeline_does_not_enable_svc_probability(self) -> None:
        pipeline = build_pipeline(pd.DataFrame({"tenure": [0, 1, 2]}))
        probability = pipeline.named_steps["classifier"].get_params().get("probability")

        self.assertIsNot(probability, True)

    def test_oof_scores_are_finite_for_each_training_row(self) -> None:
        features = pd.DataFrame({
            "tenure": np.arange(30),
            "MonthlyCharges": np.linspace(20, 90, 30),
        })
        labels = pd.Series([index % 2 for index in range(30)])

        scores = generate_oof_scores(
            features,
            labels,
            LogisticRegression(max_iter=1000),
            use_smote=False,
            n_splits=3,
        )

        self.assertEqual(scores.shape, (30,))
        self.assertTrue(np.isfinite(scores).all())

    def test_model_artifact_loads_and_scores_new_rows(self) -> None:
        rows = []
        for index in range(40):
            rows.append({
                "gender": "Female" if index % 3 else "Male",
                "tenure": index,
                "TotalCharges": str(index * 10),
                "MonthlyCharges": float(index),
            })
        features = pd.DataFrame(rows)
        labels = pd.Series([index % 2 for index in range(40)])
        pipeline = build_evaluation_pipeline(
            features,
            LogisticRegression(max_iter=1000),
            use_smote=False,
        )
        pipeline.fit(features, labels)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_model_artifact(
                pipeline,
                {"score_method": "decision_function"},
                path,
            )
            artifact = load_model_artifact(path)
            new_rows = features.iloc[:5].copy()
            result = score_customers(artifact, new_rows)
            new_rows["gender"] = " " + new_rows["gender"] + " "
            normalized_result = score_customers(artifact, new_rows)

        self.assertEqual(len(result), 5)
        self.assertEqual(result.columns.tolist(), ["prediction", "score"])
        self.assertTrue(np.isfinite(result["score"]).all())
        np.testing.assert_allclose(result["score"], normalized_result["score"])

    def test_deduplication_sensitivity_uses_train_profiles_and_grouped_cv(self) -> None:
        rows = []
        for index in range(60):
            rows.append({
                "customerID": f"C{index}",
                "gender": "Female" if index % 3 else "Male",
                "tenure": index,
                "TotalCharges": str(index * 10),
                "MonthlyCharges": float(index),
                "Churn": "Yes" if index % 2 else "No",
            })
        duplicate = rows[0].copy()
        duplicate["customerID"] = "C_DUPLICATE"
        rows.append(duplicate)

        result = compare_deduplication_sensitivity(
            pd.DataFrame(rows),
            LogisticRegression(max_iter=1000),
            use_smote=False,
        )

        self.assertEqual(result["politica"].tolist(), [
            "Sin eliminar perfiles",
            "Eliminar perfiles idénticos",
        ])
        self.assertTrue(
            np.isfinite(result[["recall_cv_media", "roc_auc_cv_media"]]).all().all()
        )
        self.assertTrue((result["perfiles_repartidos_entre_folds_naive"] >= 0).all())

    def test_operating_threshold_respects_capacity_and_costs(self) -> None:
        result = select_operating_threshold(
            [1, 1, 0, 0],
            [0.9, 0.8, 0.7, 0.1],
            capacity=2,
            contact_cost=1,
            churn_loss=10,
            retention_effectiveness=1,
            retained_value=20,
        )

        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["contacts"], 2)
        self.assertEqual(result["true_positive"], 2)

    def test_policy_economics_separates_avoided_loss_and_contact_cost(self) -> None:
        result = calculate_policy_economics(
            [1, 1, 0],
            [1, 0, 1],
            contact_cost=2,
            churn_loss=10,
            retention_effectiveness=0.2,
            retained_value=0,
        )

        self.assertEqual(result["baseline_loss"], 20)
        self.assertEqual(result["contact_cost_total"], 4)
        self.assertEqual(result["gross_loss_avoided"], 2)
        self.assertEqual(result["residual_churn_loss"], 18)
        self.assertEqual(result["expected_net_cost"], 22)
        self.assertEqual(result["net_savings_vs_no_contact"], -2)

    def test_group_metrics_report_confusion_rates_and_uncertainty(self) -> None:
        groups = pd.DataFrame({
            "gender": ["A", "A", "B", "B", "B", "B"],
        })

        result = calculate_group_metrics(
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            groups,
            min_group_size=3,
            min_class_support=2,
        ).set_index("grupo")

        self.assertEqual(result.loc["A", "falsos_positivos"], 1)
        self.assertEqual(result.loc["A", "recall"], 1.0)
        self.assertEqual(result.loc["A", "tasa_falsos_positivos"], 1.0)
        self.assertGreater(result.loc["A", "recall_ic95_superior"], 0.0)
        self.assertIn("n<3", result.loc["A", "advertencia_soporte"])
        self.assertEqual(result.loc["B", "recall"], 0.5)
        self.assertEqual(result.loc["B", "tasa_falsos_positivos"], 0.0)

    def test_group_metrics_reject_misaligned_group_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "misma cantidad de filas"):
            calculate_group_metrics(
                [1, 0],
                [1, 0],
                pd.DataFrame({"gender": ["A"]}),
            )


if __name__ == "__main__":
    unittest.main()
