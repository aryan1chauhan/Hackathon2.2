"""
Backend Module for Disaster LLM Coordinator Platform.
Provides REST and WebSocket gateway, state serialization, metrics engine, and headless runner.
"""

from backend.coordinator.simple_coordinator import SimpleCoordinator

__all__ = ["SimpleCoordinator"]
