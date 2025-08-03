#!/usr/bin/env python3
"""Debug the train/val split to see why val has no 1s."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.envs import markov_k1
import itertools

def debug_train_val_split():
    """Debug the train/val split."""
    print("DEBUG: Train/Val Split")
    print("-" * 25)
    
    # Generate buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 2000))
    
    print(f"Full buffer: {buf[:50]}...")
    print(f"Full buffer length: {len(buf)}")
    print(f"Full buffer 1-rate: {sum(buf)/len(buf):.4f}")
    
    # Simulate the train/val split from evaluate_program_calibrated
    k = 2
    N_train = 1000
    N_val = 1000
    
    # Train slice
    train_slice = buf[k:k + N_train]
    print(f"\nTrain slice: {train_slice[:20]}...")
    print(f"Train slice length: {len(train_slice)}")
    print(f"Train slice 1-rate: {sum(train_slice)/len(train_slice):.4f}")
    
    # Val slice (this is the issue!)
    val_start = k + N_train
    val_slice = buf[val_start:val_start + N_val]
    print(f"\nVal slice: {val_slice[:20]}...")
    print(f"Val slice length: {len(val_slice)}")
    print(f"Val slice 1-rate: {sum(val_slice)/len(val_slice):.4f}")
    
    # Check the actual validation loop
    print(f"\nValidation loop analysis:")
    val_losses = []
    ones = 0
    
    for i in range(val_start, min(val_start + N_val, len(buf))):
        if i >= len(buf):
            break
        
        ctx = buf[i-k:i]
        y = buf[i]
        
        val_losses.append(y)  # Just collect the y values
        ones += y
        
        if len(val_losses) <= 10:
            print(f"  i={i}, ctx={ctx}, y={y}")
    
    print(f"Validation ones: {ones}/{len(val_losses)} = {ones/len(val_losses):.4f}")
    
    # The issue: val_start might be wrong
    print(f"\nIndex analysis:")
    print(f"  k = {k}")
    print(f"  N_train = {N_train}")
    print(f"  val_start = k + N_train = {val_start}")
    print(f"  Buffer length = {len(buf)}")
    print(f"  Available for val = {len(buf) - val_start}")
    
    # Check if we're reading past the end
    if val_start >= len(buf):
        print(f"  ERROR: val_start ({val_start}) >= buffer length ({len(buf)})")
    
    return True

def main():
    debug_train_val_split()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
