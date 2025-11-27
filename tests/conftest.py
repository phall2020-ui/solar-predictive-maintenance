"""
Pytest configuration for Playwright tests.

This module provides fixtures and configuration for running
Playwright-based end-to-end tests.
"""

import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context arguments."""
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
        "ignore_https_errors": True,
    }


# Default timeout for all playwright operations
@pytest.fixture(autouse=True)
def set_default_timeout(page):
    """Set default timeout for all page operations."""
    page.set_default_timeout(30000)  # 30 seconds
    page.set_default_navigation_timeout(60000)  # 60 seconds for navigation
    yield page
