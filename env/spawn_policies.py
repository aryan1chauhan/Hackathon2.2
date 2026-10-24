"""
Victim and hazard spawn policies for the Disaster Relief Environment.
"""

import random
from typing import List, Tuple, Optional
from env.constants import Severity, DEFAULT_MAX_WAITS


class SpawnPolicy:
    """Abstract base spawn policy."""

    def spawn(self, env) -> List[Tuple[int, int]]:
        raise NotImplementedError


class DefaultSpawnPolicy(SpawnPolicy):
    """
    Standard policy: maintains an active victim count up to max_victims.
    """

    def __init__(self, spawn_probability: float = 0.2):
        self.spawn_probability = spawn_probability

    def spawn(self, env) -> List[Tuple[int, int]]:
        active_victims = [v for v in env.victims if not v.rescued and not v.died]
        if len(active_victims) >= env.max_victims:
            return []

        # If below target count, roll probability to spawn a new victim
        if random.random() < self.spawn_probability or len(active_victims) == 0:
            pos = env.get_random_empty_cell()
            if pos is not None:
                return [pos]
        return []


class RandomSpawnPolicy(SpawnPolicy):
    """
    Uniform random spawn policy.
    """

    def __init__(self, spawn_rate: float = 0.3):
        self.spawn_rate = spawn_rate

    def spawn(self, env) -> List[Tuple[int, int]]:
        active = [v for v in env.victims if not v.rescued and not v.died]
        if len(active) < env.max_victims and random.random() < self.spawn_rate:
            pos = env.get_random_empty_cell()
            if pos:
                return [pos]
        return []


class BurstSpawnPolicy(SpawnPolicy):
    """
    Simulates acute crisis waves at specific intervals.
    """

    def __init__(self, burst_interval: int = 25, burst_size: int = 3):
        self.burst_interval = burst_interval
        self.burst_size = burst_size

    def spawn(self, env) -> List[Tuple[int, int]]:
        if env.step_count > 0 and env.step_count % self.burst_interval == 0:
            spawns = []
            for _ in range(self.burst_size):
                pos = env.get_random_empty_cell()
                if pos:
                    spawns.append(pos)
            return spawns
        return []
