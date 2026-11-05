"""Shared pytest configuration."""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", max_examples=1000, deadline=None)
settings.load_profile("default")
