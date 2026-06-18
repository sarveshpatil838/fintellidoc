"""
Pytest configuration and shared fixtures.
"""

import pytest
import os


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require ANTHROPIC_API_KEY"
    )


@pytest.fixture(autouse=True)
def mock_anthropic_key(monkeypatch):
    """
    Set a dummy API key for unit tests so Settings validation passes.
    Integration tests that actually call the API should skip if key not real.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-tests")
