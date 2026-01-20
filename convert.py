#!/usr/bin/env python3
"""
Convert Hugging Face LLM models to LiteRT LM (.litertlm) format.

This script uses ai-edge-torch to convert PyTorch-based LLM models from
Hugging Face to LiteRT format and packages them as .litertlm files with
tokenizer and metadata for deployment on mobile and edge devices.
"""

import argparse
import sys
import tempfile
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import ai_edge_torch
    from ai_edge_torch.generative.quantize import quant_recipes
    from ai_edge_torch.generative.utilities import litertlm_builder
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


def convert_hf_to_litertlm(
    model_name: str,
    output_path: str,
    quantize: bool = False,
    max_seq_length: int = 512,
    build_litertlm: bool = True,
) -> None:
    """
    Convert a Hugging Face model to LiteRT LM format.
    
    Args:
        model_name: Name or path of the Hugging Face model
        output_path: Path where the converted model will be saved
        quantize: Whether to apply quantization (default: False)
        max_seq_length: Maximum sequence length for the model (default: 512)
        build_litertlm: Whether to build a .litertlm file (default: True)
    """
    print(f"Loading model '{model_name}' from Hugging Face...")
    
    try:
        # Load tokenizer and model
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False,  # Security: Don't execute remote code
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,  # Security: Don't execute remote code
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
        # Pass both input_ids and attention_mask for proper inference
        edge_model = ai_edge_torch.convert(
            model,
            (sample_input["input_ids"], sample_input["attention_mask"])
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
        
        # Determine if we're building a .litertlm file
        if build_litertlm and output_path.endswith('.litertlm'):
            # First save as TFLite
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_tflite = Path(temp_dir) / "model.tflite"
                print(f"Saving intermediate TFLite model...")
                edge_model.export(str(temp_tflite))
                
                # Save tokenizer for litertlm
                tokenizer_path = Path(temp_dir) / "tokenizer"
                tokenizer.save_pretrained(str(tokenizer_path))
                
                # Build .litertlm file
                print(f"Building .litertlm package...")
                if not litertlm_builder.is_litertlm_builder_available():
                    print("Warning: litertlm_builder not available. Saving as .tflite instead.")
                    # Fall back to saving as tflite
                    edge_model.export(output_path.replace('.litertlm', '.tflite'))
                    print(f"✓ Conversion successful!")
                    print(f"  Model saved to: {output_path.replace('.litertlm', '.tflite')}")
                else:
                    litertlm_builder.build_litertlm(
                        tflite_model_path=str(temp_tflite),
                        workdir=temp_dir,
                        output_path=str(output_dir),
                        context_length=max_seq_length,
                        hf_tokenizer_model_path=str(tokenizer_path),
                        llm_model_type='generic',
                    )
                    print(f"✓ Conversion successful!")
                    print(f"  LiteRT LM model saved to: {output_path}")
        else:
            # Save as TFLite only
            if not output_path.endswith('.tflite'):
                output_path = output_path.replace('.litertlm', '.tflite')
            
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face LLM models to LiteRT LM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to .litertlm format (recommended)
  python convert.py --model gpt2 --output models/gpt2.litertlm
  
  # Convert to .tflite format only
  python convert.py --model gpt2 --output models/gpt2.tflite --tflite-only
  
  # Convert with quantization
  python convert.py --model gpt2 --output models/gpt2.litertlm --quantize
  
  # Convert with custom max sequence length
  python convert.py --model gpt2 --output models/gpt2.litertlm --max-seq-length 1024
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
        help="Output path for the converted model (e.g., 'output/model.litertlm' or 'output/model.tflite')"
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
        help="Maximum sequence length for the model (default: 512, must be between 1 and 8192)"
    )
    
    parser.add_argument(
        "--tflite-only",
        action="store_true",
        help="Only generate TFLite file, don't build .litertlm package"
    )
    
    args = parser.parse_args()
    
    # Validate max_seq_length
    if args.max_seq_length < 1 or args.max_seq_length > 8192:
        parser.error("--max-seq-length must be between 1 and 8192")
    
    # Determine output format
    build_litertlm_file = not args.tflite_only and (
        args.output.endswith('.litertlm') or 
        not args.output.endswith('.tflite')
    )
    
    # Ensure proper extension
    if build_litertlm_file and not args.output.endswith('.litertlm'):
        args.output = args.output.rsplit('.', 1)[0] + '.litertlm' if '.' in args.output else args.output + '.litertlm'
    
    print("=" * 60)
    print("HuggingFace to LiteRT LM Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Format: {'.litertlm package' if build_litertlm_file else '.tflite only'}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    print("=" * 60)
    print()
    
    convert_hf_to_litertlm(
        model_name=args.model,
        output_path=args.output,
        quantize=args.quantize,
        max_seq_length=args.max_seq_length,
        build_litertlm=build_litertlm_file,
    )


if __name__ == "__main__":
    main()
