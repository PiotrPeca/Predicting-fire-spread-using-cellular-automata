"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_grid_size():
    """Provide a standard grid size for tests."""
    return (10, 10)
