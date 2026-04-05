"""Tests for latent_stats_debug helpers."""

from __future__ import annotations

import torch

from services.latent_stats_debug import install_latent_stats_hooks, latent_stat_dict


def test_latent_stat_dict_finite_cpu() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    d = latent_stat_dict(t)
    assert d["shape"] == (3,)
    assert d["numel"] == 3
    assert d["finite_frac"] == 1.0
    assert d["min"] == 1.0
    assert d["max"] == 3.0


def test_latent_stat_dict_with_nan() -> None:
    t = torch.tensor([1.0, float("nan"), 3.0])
    d = latent_stat_dict(t)
    assert d["finite_frac"] < 1.0
    assert d["min"] == 1.0
    assert d["max"] == 3.0


def test_install_latent_stats_hooks_idempotent() -> None:
    install_latent_stats_hooks()
    install_latent_stats_hooks()
