#!/usr/bin/env python3
"""
Example usage of the HuggingFace to LiteRT converter.

This script demonstrates how to use the convert.py module programmatically.
"""

from convert import convert_hf_to_litertlm

# Example 1: Convert GPT-2 to LiteRT
print("Example 1: Converting GPT-2 to LiteRT")
try:
    convert_hf_to_litertlm(
        model_name="gpt2",
        output_path="models/gpt2.tflite",
        quantize=False,
        max_seq_length=512,
    )
except Exception as e:
    print(f"Note: This example requires dependencies to be installed. Error: {e}")

# Example 2: Convert with quantization
print("\nExample 2: Converting with quantization")
try:
    convert_hf_to_litertlm(
        model_name="gpt2",
        output_path="models/gpt2_quantized.tflite",
        quantize=True,
        max_seq_length=512,
    )
except Exception as e:
    print(f"Note: This example requires dependencies to be installed. Error: {e}")
