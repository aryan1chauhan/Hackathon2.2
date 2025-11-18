import random
from typing import Tuple, Optional
from env.constants import Severity


class SpawnPolicy:
    """Base class for victim spawning policies"""
    
    def should_spawn(self, current_step: int, active_victims: int, max_victims: int) -> bool:
        """Determine if a victim should spawn this step"""
        raise NotImplementedError
    
    def get_spawn_params(self, current_step: int) -> Tuple[Severity, int]:
        """Get severity and max_wait for new victim"""
        raise NotImplementedError


class RandomSpawnPolicy(SpawnPolicy):
    """Random victim spawning with configurable rate"""
    
    def __init__(self, spawn_rate: float = 0.15, seed: Optional[int] = None):
        self.spawn_rate = spawn_rate
        self._random = random.Random(seed)
    
    def should_spawn(self, current_step: int, active_victims: int, max_victims: int) -> bool:
        if active_victims >= max_victims:
            return False
        return self._random.random() < self.spawn_rate
    
    def get_spawn_params(self, current_step: int) -> Tuple[Severity, int]:
        """Random severity with weighted probabilities"""
        severity_roll = self._random.random()
        if severity_roll < 0.3:
            severity = Severity.CRITICAL
            max_wait = self._random.randint(40, 60)
        elif severity_roll < 0.6:
            severity = Severity.SERIOUS
            max_wait = self._random.randint(70, 90)
        else:
            severity = Severity.STABLE
            max_wait = self._random.randint(100, 140)
        
        return severity, max_wait


class BurstSpawnPolicy(SpawnPolicy):
    """
    Simulates disaster waves - periods of high victim spawning.
    Useful for testing agent coordination under stress.
    """
    
    def __init__(self, burst_interval: int = 100, burst_duration: int = 20, 
                 burst_rate: float = 0.5, normal_rate: float = 0.1):
        self.burst_interval = burst_interval
        self.burst_duration = burst_duration
        self.burst_rate = burst_rate
        self.normal_rate = normal_rate
    
    def should_spawn(self, current_step: int, active_victims: int, max_victims: int) -> bool:
        if active_victims >= max_victims:
            return False
        
        # Check if in burst period
        cycle_position = current_step % self.burst_interval
        in_burst = cycle_position < self.burst_duration
        
        rate = self.burst_rate if in_burst else self.normal_rate
        return random.random() < rate
    
    def get_spawn_params(self, current_step: int) -> Tuple[Severity, int]:
        # During burst, more critical victims
        cycle_position = current_step % self.burst_interval
        in_burst = cycle_position < self.burst_duration
        
        if in_burst:
            # Higher chance of critical victims during burst
            severity_roll = random.random()
            if severity_roll < 0.5:
                severity = Severity.CRITICAL
                max_wait = random.randint(40, 60)
            elif severity_roll < 0.8:
                severity = Severity.SERIOUS
                max_wait = random.randint(70, 90)
            else:
                severity = Severity.STABLE
                max_wait = random.randint(100, 140)
        else:
            # Normal distribution
            severity_roll = random.random()
            if severity_roll < 0.2:
                severity = Severity.CRITICAL
                max_wait = random.randint(40, 60)
            elif severity_roll < 0.5:
                severity = Severity.SERIOUS
                max_wait = random.randint(70, 90)
            else:
                severity = Severity.STABLE
                max_wait = random.randint(100, 140)
        
        return severity, max_wait


class ConstantSpawnPolicy(SpawnPolicy):
    """Spawn victims at constant rate for predictable testing"""
    
    def __init__(self, spawn_every_n_steps: int = 10):
        self.spawn_every_n_steps = spawn_every_n_steps
    
    def should_spawn(self, current_step: int, active_victims: int, max_victims: int) -> bool:
        if active_victims >= max_victims:
            return False
        return current_step % self.spawn_every_n_steps == 0
    
    def get_spawn_params(self, current_step: int) -> Tuple[Severity, int]:
        # Cycle through severities predictably
        cycle = (current_step // self.spawn_every_n_steps) % 3
        if cycle == 0:
            return Severity.CRITICAL, 50
        elif cycle == 1:
            return Severity.SERIOUS, 80
        else:
            return Severity.STABLE, 120