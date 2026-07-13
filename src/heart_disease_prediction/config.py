"""Configuration objects for heart disease prediction experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for a heart disease prediction experiment.

    Attributes:
        data_path: Path to the input CSV file.
        target_column: Name of the binary target column.
        random_state: Seed used for reproducible splits and models.
        test_size: Fraction reserved for final testing.
        validation_size: Fraction of training data reserved for validation.
    """

    data_path: Path
    target_column: str
    random_state: int
    test_size: float
    validation_size: float

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Build configuration from environment variables.

        Returns:
            ExperimentConfig: Validated runtime configuration.

        Raises:
            ValueError: If required environment variables are missing.
        """

        data_path = os.getenv("HEART_DATA_PATH")
        if not data_path:
            raise ValueError(
                "Set HEART_DATA_PATH to the cardiovascular dataset CSV path."
            )

        return cls(
            data_path=Path(data_path),
            target_column=os.getenv("HEART_TARGET_COLUMN", "cardio"),
            random_state=int(os.getenv("HEART_RANDOM_STATE", "42")),
            test_size=float(os.getenv("HEART_TEST_SIZE", "0.20")),
            validation_size=float(os.getenv("HEART_VALIDATION_SIZE", "0.20")),
        )
