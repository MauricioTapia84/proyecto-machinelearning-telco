import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import deduplicate_customer_profiles
from src.pipeline import (
    PCAKMeansFeatures,
    build_evaluation_pipeline,
    prepare_dataset,
)


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


if __name__ == "__main__":
    unittest.main()
