#!/usr/bin/env python3
"""
Convert Hugging Face LLM models to LiteRT (TensorFlow Lite) format.

This script uses ai-edge-torch to convert PyTorch-based LLM models from
Hugging Face to LiteRT format for deployment on mobile and edge devices.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import ai_edge_torch
    from ai_edge_torch.generative.layers import attention
    from ai_edge_torch.generative.quantize import quant_recipes
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


def convert_hf_to_litertlm(
    model_name: str,
    output_path: str,
    quantize: bool = False,
    max_seq_length: int = 512,
) -> None:
    """
    Convert a Hugging Face model to LiteRT format.
    
    Args:
        model_name: Name or path of the Hugging Face model
        output_path: Path where the converted model will be saved
        quantize: Whether to apply quantization (default: False)
        max_seq_length: Maximum sequence length for the model (default: 512)
    """
    print(f"Loading model '{model_name}' from Hugging Face...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        print(f"Model loaded successfully. Converting to LiteRT format...")
        
        # Prepare sample input for tracing
        sample_input = tokenizer(
            "Hello, how are you?",
            return_tensors="pt",
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
        )
        
        # Convert to LiteRT using ai-edge-torch
        # Note: The exact conversion API depends on the ai-edge-torch version
        # This is a general approach that may need adjustment based on model architecture
        edge_model = ai_edge_torch.convert(
            model,
            (sample_input["input_ids"],)
        )
        
        # Apply quantization if requested
        if quantize:
            print("Applying quantization...")
            # Apply dynamic range quantization
            edge_model = ai_edge_torch.quantize(
                edge_model,
                quant_recipes.dynamic_range_quantization()
            )
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the converted model
        print(f"Saving converted model to '{output_path}'...")
        edge_model.export(output_path)
        
        # Save tokenizer for reference
        tokenizer_output_path = str(Path(output_path).parent / "tokenizer")
        tokenizer.save_pretrained(tokenizer_output_path)
        
        print(f"✓ Conversion successful!")
        print(f"  Model saved to: {output_path}")
        print(f"  Tokenizer saved to: {tokenizer_output_path}")
        
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model name is correct and accessible on Hugging Face")
        print("2. Check that you have enough memory for the model")
        print("3. Some models may require special handling - check ai-edge-torch documentation")
        sys.exit(1)


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face LLM models to LiteRT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a small model
  python convert.py --model gpt2 --output models/gpt2.tflite
  
  # Convert with quantization
  python convert.py --model gpt2 --output models/gpt2.tflite --quantize
  
  # Convert with custom max sequence length
  python convert.py --model gpt2 --output models/gpt2.tflite --max-seq-length 1024
        """
    )
    
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Name or path of the Hugging Face model (e.g., 'gpt2', 'facebook/opt-125m')"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the converted LiteRT model (e.g., 'output/model.tflite')"
    )
    
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Apply quantization to reduce model size (may affect accuracy)"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for the model (default: 512)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HuggingFace to LiteRT Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    print("=" * 60)
    print()
    
    convert_hf_to_litertlm(
        model_name=args.model,
        output_path=args.output,
        quantize=args.quantize,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
