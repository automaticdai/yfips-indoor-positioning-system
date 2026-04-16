"""Smoke tests: ensure pure-logic modules import without side effects."""

import importlib


def test_yfips_package_imports():
    importlib.import_module("yfips")


def test_pure_logic_modules_import():
    for name in (
        "yfips.config",
        "yfips.tracker",
        "yfips.kalman_tracker",
        "yfips.image_detector",
        "yfips.ros_publisher",
    ):
        importlib.import_module(name)
