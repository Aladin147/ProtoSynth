"""
ProtoSynth Stream Environments

This module implements various binary stream generators for testing
prediction algorithms. Each environment provides a deterministic
sequence of 0/1 bits based on different underlying patterns.
"""

import random
import itertools
from typing import Iterator, List, Dict, Tuple, Union


def periodic(pattern_bits: Union[List[int], str], seed: int = 0) -> Iterator[int]:
    """
    Generate a repeating pattern of bits.
    
    Args:
        pattern_bits: Pattern to repeat (list of 0/1 or binary string)
        seed: Random seed (for consistency with other generators)
        
    Yields:
        int: Next bit in the repeating pattern (0 or 1)
        
    Example:
        >>> list(itertools.islice(periodic([1, 0, 1]), 6))
        [1, 0, 1, 1, 0, 1]
    """
    # Convert string to list if needed
    if isinstance(pattern_bits, str):
        pattern_bits = [int(b) for b in pattern_bits]
    
    if not pattern_bits:
        raise ValueError("Pattern cannot be empty")
    
    # Validate pattern contains only 0/1
    for bit in pattern_bits:
        if bit not in [0, 1]:
            raise ValueError(f"Pattern must contain only 0/1, got {bit}")
    
    i = 0
    while True:
        yield pattern_bits[i % len(pattern_bits)]
        i += 1


def periodic_k4(seed: int = 0) -> Iterator[int]:
    """Generate periodic pattern with period 4 for k=4 benchmark."""
    return periodic([1, 0, 1, 1], seed)


def markov_k1(p_stay: float = 0.8, seed: int = 0) -> Iterator[int]:
    """
    Generate first-order Markov chain for k=1 benchmark.

    Args:
        p_stay: Probability of staying in same state
        seed: Random seed

    Yields:
        int: Next bit based on Markov transition
    """
    rnd = random.Random(seed)
    s = rnd.randrange(2)  # current bit

    while True:
        yield s
        # Next state: stay with p_stay, flip otherwise
        if rnd.random() < p_stay:
            s = s  # stay
        else:
            s ^= 1  # flip

def markov_k2(p_stay: float = 0.8, seed: int = 0) -> Iterator[int]:
    """
    Generate first-order Markov chain for k=1 benchmark.
    Note: Despite the name 'k2', this is actually k=1 (first-order).
    Use markov_k1 directly for clarity.
    """
    return markov_k1(p_stay, seed)

def check_transitions(gen_factory, steps: int = 10000):
    """Check transition probabilities of a generator."""
    g = gen_factory()
    prev = next(g)
    counts = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}

    for _ in range(steps - 1):
        cur = next(g)
        counts[(prev, cur)] += 1
        prev = cur

    total = steps - 1
    stay = (counts[(0,0)] + counts[(1,1)]) / total
    flip = 1 - stay

    return counts, stay, flip


def k_order_markov(k: int, trans: Dict[Tuple[int, ...], float], seed: int = 0) -> Iterator[int]:
    """
    Generate bits using a k-order Markov chain.

    Args:
        k: Order of the Markov chain (context length)
        trans: Transition probabilities {context_tuple: prob_of_1}
        seed: Random seed for deterministic generation

    Yields:
        int: Next bit (0 or 1) based on Markov transitions

    Example:
        >>> trans = {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.8, (1, 1): 0.2}
        >>> list(itertools.islice(k_order_markov(2, trans, seed=42), 5))
        [0, 0, 0, 1, 1]
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    rng = random.Random(seed)

    # Initialize context with zeros
    ctx = [0] * k

    while True:
        key = tuple(ctx)
        p1 = trans.get(key, 0.5)  # Default to 0.5 if context not in table

        # Clamp probability to valid range
        p1 = max(0.0, min(1.0, p1))

        # Generate next bit
        bit = 1 if rng.random() < p1 else 0
        yield bit

        # Update context (sliding window)
        if k > 0:
            ctx = (ctx + [bit])[-k:]


def arith_prog(a: int, d: int, mod: int, seed: int = 0) -> Iterator[int]:
    """
    Generate bits from arithmetic progression (a + n*d) % mod, emit LSB.
    
    Args:
        a: Starting value
        d: Common difference
        mod: Modulus
        seed: Random seed (for consistency, not used in deterministic sequence)
        
    Yields:
        int: LSB of (a + n*d) % mod for n = 0, 1, 2, ...
        
    Example:
        >>> list(itertools.islice(arith_prog(1, 3, 8), 8))
        [1, 0, 1, 0, 1, 0, 1, 0]  # LSBs of [1, 4, 7, 2, 5, 0, 3, 6]
    """
    if mod <= 0:
        raise ValueError("Modulus must be positive")
    
    n = 0
    while True:
        value = (a + n * d) % mod
        bit = value & 1  # Extract LSB
        yield bit
        n += 1


def noisy(base_generator: Iterator[int], p_flip: float, seed: int = 0) -> Iterator[int]:
    """
    Add Bernoulli noise to a base bit stream.

    Args:
        base_generator: Base stream generator
        p_flip: Probability of flipping each bit
        seed: Random seed for noise generation

    Yields:
        int: Noisy bit (0 or 1)

    Example:
        >>> base = periodic([1, 0])
        >>> noisy_stream = noisy(base, 0.1, seed=42)
        >>> list(itertools.islice(noisy_stream, 6))
        [1, 0, 1, 0, 1, 1]  # Some bits may be flipped
    """
    if not (0.0 <= p_flip <= 1.0):
        raise ValueError("p_flip must be in [0, 1]")

    rng = random.Random(seed)

    for bit in base_generator:
        if rng.random() < p_flip:
            bit = 1 - bit  # Flip the bit
        yield bit


# Convenience functions for common patterns
def alternating(seed: int = 0) -> Iterator[int]:
    """Generate alternating 0, 1, 0, 1, ... pattern."""
    return periodic([0, 1], seed)


def constant(value: int, seed: int = 0) -> Iterator[int]:
    """Generate constant stream of 0s or 1s."""
    if value not in [0, 1]:
        raise ValueError("Value must be 0 or 1")
    return periodic([value], seed)


def random_bits(p: float = 0.5, seed: int = 0) -> Iterator[int]:
    """Generate random bits with probability p of emitting 1."""
    # Use 0-order Markov chain (no context)
    return k_order_markov(0, {(): p}, seed)


# Predefined transition tables for testing
SIMPLE_MARKOV_TABLES = {
    "alternating": {
        (0,): 1.0,  # After 0, always emit 1
        (1,): 0.0,  # After 1, always emit 0
    },
    
    "sticky": {
        (0,): 0.1,  # After 0, usually stay 0
        (1,): 0.9,  # After 1, usually stay 1
    },
    
    "anti_sticky": {
        (0,): 0.9,  # After 0, usually flip to 1
        (1,): 0.1,  # After 1, usually flip to 0
    },
    
    "second_order_xor": {
        (0, 0): 0.0,  # 00 -> 0 (even parity)
        (0, 1): 1.0,  # 01 -> 1 (odd parity)
        (1, 0): 1.0,  # 10 -> 1 (odd parity)
        (1, 1): 0.0,  # 11 -> 0 (even parity)
    }
}


def create_environment(env_type: str, **kwargs) -> Iterator[int]:
    """
    Factory function to create stream environments.
    
    Args:
        env_type: Type of environment ('periodic', 'markov', 'arith', 'noisy', 'random')
        **kwargs: Environment-specific parameters
        
    Returns:
        Iterator[int]: Bit stream generator
    """
    if env_type == "periodic":
        pattern = kwargs.get("pattern", [1, 0])
        seed = kwargs.get("seed", 0)
        return periodic(pattern, seed)
    
    elif env_type == "markov":
        k = kwargs.get("k", 1)
        trans = kwargs.get("trans", SIMPLE_MARKOV_TABLES["alternating"])
        seed = kwargs.get("seed", 0)
        return k_order_markov(k, trans, seed)
    
    elif env_type == "arith":
        a = kwargs.get("a", 1)
        d = kwargs.get("d", 3)
        mod = kwargs.get("mod", 8)
        seed = kwargs.get("seed", 0)
        return arith_prog(a, d, mod, seed)
    
    elif env_type == "noisy":
        base_type = kwargs.get("base_type", "periodic")
        base_kwargs = kwargs.get("base_kwargs", {"pattern": [1, 0]})
        p_flip = kwargs.get("p_flip", 0.1)
        seed = kwargs.get("seed", 0)
        
        base = create_environment(base_type, **base_kwargs)
        return noisy(base, p_flip, seed)
    
    elif env_type == "random":
        p = kwargs.get("p", 0.5)
        seed = kwargs.get("seed", 0)
        return random_bits(p, seed)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def get_stream_factory(env_name: str):
    """Get a stream factory function for the given environment name."""
    if env_name == "periodic_k4":
        return lambda: periodic([1, 0, 1, 0], seed=42)
    elif env_name == "markov_k2":
        return lambda: markov_k1(p_stay=0.8, seed=42)
    elif env_name == "alternating":
        return lambda: periodic([0, 1], seed=42)
    elif env_name == "random":
        return lambda: random_bits(p=0.5, seed=42)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
