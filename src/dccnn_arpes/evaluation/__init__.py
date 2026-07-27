"""Quantitative and physical-fidelity evaluation for ARPES cuts."""

from .baselines import gaussian_baseline, median_baseline
from .metrics import evaluate_pair, physical_features
from .report import EvaluationCase, generate_evaluation_report

__all__ = [
    "EvaluationCase",
    "evaluate_pair",
    "gaussian_baseline",
    "generate_evaluation_report",
    "median_baseline",
    "physical_features",
]
