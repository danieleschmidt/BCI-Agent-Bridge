"""API module for BCI-Agent-Bridge."""

from fastapi import FastAPI

def create_app() -> FastAPI:
    """Create FastAPI application."""
    from ..__main__ import create_application
    return create_application()

__all__ = ["create_app"]