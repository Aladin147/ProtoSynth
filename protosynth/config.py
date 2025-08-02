"""
ProtoSynth Configuration

This module provides configuration settings for mutation rates,
verification limits, and other system parameters.
"""

import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class MutationConfig:
    """Configuration for mutation operations."""
    mutation_rate: float = 0.15
    const_perturb_delta: int = 10
    max_mutation_attempts: int = 10


@dataclass
class VerifierConfig:
    """Configuration for AST verification."""
    max_depth: int = 10
    max_nodes: int = 100


@dataclass
class InterpreterConfig:
    """Configuration for the Lisp interpreter."""
    max_recursion_depth: int = 10
    max_steps: int = 100
    timeout_seconds: float = 1.0


@dataclass
class ProtoSynthConfig:
    """Main configuration for ProtoSynth system."""
    mutation: MutationConfig = None
    verifier: VerifierConfig = None
    interpreter: InterpreterConfig = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.mutation is None:
            self.mutation = MutationConfig()
        if self.verifier is None:
            self.verifier = VerifierConfig()
        if self.interpreter is None:
            self.interpreter = InterpreterConfig()


# Global configuration instance
_config: Optional[ProtoSynthConfig] = None


def get_config() -> ProtoSynthConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ProtoSynthConfig()
    return _config


def set_config(config: ProtoSynthConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def setup_logging(level: str = "INFO") -> None:
    """Setup logging for ProtoSynth."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Setup default logging
setup_logging(get_config().log_level)

# Create logger for this module
logger = logging.getLogger(__name__)
