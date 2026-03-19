"""Basic tests to ensure all packages are properly installed."""

import importlib

import pytest


@pytest.mark.parametrize("package_name", ["commons", "arxiv_curator", "learning_buddy"])
def test_package_import(package_name: str) -> None:
    """Test that each package can be imported."""
    module = importlib.import_module(package_name)
    assert module is not None


@pytest.mark.parametrize("package_name", ["commons", "arxiv_curator", "learning_buddy"])
def test_version_exists(package_name: str) -> None:
    """Test that each package has a version attribute."""
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
