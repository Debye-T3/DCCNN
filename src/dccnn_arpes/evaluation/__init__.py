"""Quantitative and physical-fidelity evaluation for ARPES cuts."""

from .baselines import gaussian_baseline, median_baseline
from .metrics import evaluate_pair, physical_features
from .real_pairs import compare_pair, count_rate_normalize, effective_exposure, orient_pair
from .report import EvaluationCase, generate_evaluation_report

__all__ = [
    "EvaluationCase",
    "compare_pair",
    "count_rate_normalize",
    "effective_exposure",
    "evaluate_pair",
    "gaussian_baseline",
    "generate_evaluation_report",
    "median_baseline",
    "orient_pair",
    "physical_features",
]
