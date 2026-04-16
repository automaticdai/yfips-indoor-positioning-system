"""Tests for AprilTag detector option mapping from config."""

from yfips.detection import apriltag_options_kwargs


def test_empty_config_yields_empty_kwargs():
    assert apriltag_options_kwargs({}) == {}


def test_missing_block_yields_empty():
    assert apriltag_options_kwargs({"mode": "apriltag"}) == {}


def test_family_remapped_to_families():
    out = apriltag_options_kwargs({"apriltag_mode": {"family": "tag25h9"}})
    assert out == {"families": "tag25h9"}


def test_passes_nthreads_quad_decimate_refine_edges():
    cfg = {"apriltag_mode": {
        "nthreads": 8,
        "quad_decimate": 2.0,
        "refine_edges": False,
    }}
    out = apriltag_options_kwargs(cfg)
    assert out == {
        "nthreads": 8,
        "quad_decimate": 2.0,
        "refine_edges": False,
    }


def test_unknown_keys_ignored():
    cfg = {"apriltag_mode": {"family": "tag36h11", "nonsense": 42}}
    out = apriltag_options_kwargs(cfg)
    assert out == {"families": "tag36h11"}
