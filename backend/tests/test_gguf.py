"""Tests for GGUF mode: specs, paths, and stub pipeline behavior."""

from __future__ import annotations

import pytest

from services.fast_video_pipeline.gguf_stub_pipeline import GGUFStubFastVideoPipeline


def test_gguf_stub_create_returns_stub():
    pipeline = GGUFStubFastVideoPipeline.create("", None, "", "cpu")
    assert pipeline is not None
    assert pipeline.pipeline_kind == "fast"


def test_gguf_stub_generate_raises():
    pipeline = GGUFStubFastVideoPipeline.create("", None, "", "cpu")
    with pytest.raises(RuntimeError, match="GGUF inference is not yet implemented"):
        pipeline.generate(
            prompt="test",
            seed=1,
            height=256,
            width=384,
            num_frames=9,
            frame_rate=8,
            images=[],
            output_path="/tmp/out.mp4",
        )


def test_gguf_stub_warmup_raises():
    pipeline = GGUFStubFastVideoPipeline.create("", None, "", "cpu")
    with pytest.raises(RuntimeError, match="GGUF inference is not yet implemented"):
        pipeline.warmup(output_path="/tmp/warmup.mp4")


def test_gguf_stub_compile_transformer_no_op():
    pipeline = GGUFStubFastVideoPipeline.create("", None, "", "cpu")
    pipeline.compile_transformer()  # no raise
