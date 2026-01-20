#!/usr/bin/env python3
"""
Example usage of the auto-detect converter.

This script demonstrates how to use the auto-detect converter
programmatically or as an interactive tool.
"""

# Example 1: Interactive usage (recommended)
# Simply run: python convert_autodetect.py
# Then enter a model ID like: google/gemma-2b-it

# Example 2: Programmatic usage
# You can import and use the functions from convert_autodetect.py

from convert_autodetect import ARCH_CONFIG, main
import os
import sys

# Display supported architectures
print("=== Auto-Detect Converter Example ===\n")
print("Supported Model Architectures:")
for arch, config in ARCH_CONFIG.items():
    print(f"  - {arch}: {config['import_path']}")
    print(f"    Start token: {config['start_token']}")
    print(f"    Stop tokens: {config['stop_tokens']}")
    print()

print("\nRecommended Models to Try:")
print("  - google/gemma-2b-it (Gemma2)")
print("  - meta-llama/Llama-2-7b-chat-hf (Llama)")
print("  - microsoft/phi-2 (Phi)")
print("  - Qwen/Qwen2-0.5B (Qwen2)")
print()

print("To convert a model, run:")
print("  python convert_autodetect.py")
print()
print("Then enter the Hugging Face model ID when prompted.")
print()

# Note: The actual conversion requires the model to be downloaded
# and proper dependencies installed. Uncomment the following line
# to run the interactive converter:
# main()
