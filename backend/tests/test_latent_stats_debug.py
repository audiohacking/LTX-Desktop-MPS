"""Tests for latent_stats_debug helpers."""

from __future__ import annotations

import pytest
import torch

import services.latent_stats_debug as latent_stats_debug
from services.latent_stats_debug import latent_stat_dict


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


@pytest.mark.parametrize("via", ("env", "flag"))
def test_z_latent_stats_hooks_install_idempotent(monkeypatch: pytest.MonkeyPatch, via: str) -> None:
    """Last in file: patches distilled; must uninstall for other tests."""
    latent_stats_debug.uninstall_latent_stats_hooks()
    latent_stats_debug._hooks_installed = False
    latent_stats_debug._saved_denoise = None
    latent_stats_debug._saved_decode = None
    try:
        if via == "env":
            monkeypatch.setenv("LTX_DEBUG_LATENT_STATS", "1")
            monkeypatch.setattr(latent_stats_debug, "LATENT_STATS_DEBUG_ENABLED", False)
        else:
            monkeypatch.delenv("LTX_DEBUG_LATENT_STATS", raising=False)
            monkeypatch.setattr(latent_stats_debug, "LATENT_STATS_DEBUG_ENABLED", True)

        latent_stats_debug.install_latent_stats_hooks()
        latent_stats_debug.install_latent_stats_hooks()
        assert latent_stats_debug._hooks_installed
    finally:
        latent_stats_debug.uninstall_latent_stats_hooks()
