"""Unit checks for the frozen Figure 3 mechanism-analysis utilities."""

from __future__ import annotations

import math

import torch

from scripts.mrage_figure3_common import (
    CONTEXT_ORDER,
    coefficient_rows,
    jensen_shannon_divergence,
    normalized_coefficients,
    primary_subset,
    routing_rows,
    routing_stats,
    top2_statistics,
)
from src.models.mrage import SharedExpertBank


def test_effective_coefficients_match_the_model_bank():
    bank = SharedExpertBank(hidden_dim=16, bottleneck_dim=8, dropout=0.0, eta=0.1)
    bank.theta.data.copy_(torch.randn_like(bank.theta))
    for max_order in (1, 3):
        # The bank returns all four columns with higher orders masked to zero;
        # forward() slices to the first max_order+1 columns.
        expected = bank.coefficients(max_order, bank.theta.dtype)[:, : max_order + 1]
        actual = normalized_coefficients(bank.theta.detach(), max_order)
        assert torch.allclose(expected, actual, atol=1.0e-7)
        assert actual.shape == (4, max_order + 1)


def test_coefficient_rows_cover_physical_and_semantic_orders():
    theta = torch.randn(4, 4)
    rows = coefficient_rows(
        dataset="Synthetic",
        training_seed=42,
        checkpoint="synthetic.pt",
        theta=theta,
    )
    physical = [row for row in rows if row["relation"] == "physical"]
    semantic = [row for row in rows if row["relation"] == "semantic"]
    assert len(physical) == 4 * 4
    assert len(semantic) == 4 * 2
    assert {row["order"] for row in physical} == {0, 1, 2, 3}
    assert {row["order"] for row in semantic} == {0, 1}


def test_top2_statistics_weight_and_frequency_sums():
    torch.manual_seed(7)
    logits = torch.randn(257, 4)
    stats = top2_statistics(logits)
    assert stats["mean_top2_weight"].shape == (4,)
    assert stats["top2_selection_frequency"].shape == (4,)
    assert math.isclose(float(stats["mean_top2_weight"].sum()), 1.0, abs_tol=1.0e-6)
    assert math.isclose(
        float(stats["top2_selection_frequency"].sum()), 2.0, abs_tol=1.0e-6
    )
    assert 1 <= int(stats["n_top2_combinations"]) <= 12
    assert stats["node_route_std"].shape == (4,)
    assert isinstance(stats["most_common_top2_combo_label"], str)


def test_routing_stats_and_rows_use_primary_subset_names():
    torch.manual_seed(11)
    logits = torch.randn(64, 4)
    support = torch.ones(64, dtype=torch.bool)
    support[:8] = False
    cross = torch.zeros(64, 8)
    cross[:16, 0] = 1.0
    physical = routing_stats(logits, support, cross)
    assert physical["primary"] is not None
    assert physical["cross"] is not None
    assert physical["n_active"] == int(support.sum())
    assert physical["cross_nonzero_count"] == 8
    assert primary_subset("private_text_physical") == "physical_support"
    assert primary_subset("private_text_semantic") == "semantic_support"
    records = {
        context: routing_stats(
            logits,
            support,
            cross if context.startswith("collaborative") else None,
        )
        for context in CONTEXT_ORDER
    }
    pattern_rows, diagnostic_rows, distributions = routing_rows(
        dataset="Synthetic",
        training_seed=42,
        checkpoint="synthetic.pt",
        records=records,
    )
    assert len(pattern_rows) == (len(CONTEXT_ORDER) + 2) * 4
    assert len(diagnostic_rows) == len(CONTEXT_ORDER) + 2
    assert all(
        row["subset"] in {"physical_support", "semantic_support", "cross_nonzero"}
        for row in pattern_rows
    )
    assert "private_text_semantic:semantic_support" in distributions
    assert "private_text_physical:physical_support" in distributions
    assert "collaborative_text_physical:cross_nonzero" in distributions


def test_jensen_shannon_divergence_is_zero_for_identical_and_symmetric():
    first = torch.tensor([0.1, 0.2, 0.3, 0.4])
    second = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert math.isclose(jensen_shannon_divergence(first, first), 0.0, abs_tol=1.0e-9)
    forward = jensen_shannon_divergence(first, second)
    backward = jensen_shannon_divergence(second, first)
    assert math.isclose(forward, backward, rel_tol=1.0e-6, abs_tol=1.0e-9)
    assert forward > 0.0
