#!/usr/bin/env python3
"""
Convert Hugging Face LLM models to LiteRT LM (.litertlm) format.

This script uses ai-edge-torch to convert PyTorch-based LLM models from
Hugging Face to LiteRT format and packages them as .litertlm files with
tokenizer and metadata for deployment on mobile and edge devices.
"""

import argparse
import shutil
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


class ModelWrapper(torch.nn.Module):
    """
    Wrapper to extract only logits from model output.
    
    This wrapper ensures that the model returns only tensors (logits)
    instead of the full output tuple that may contain unsupported types
    like DynamicCache, which causes issues with torch.export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        """
        Forward pass that returns only logits.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Logits tensor only (no cache or other objects)
        """
        # Call model with use_cache=False to prevent DynamicCache generation
        # Most transformer models support these parameters, but we handle gracefully
        try:
            outputs = self.model(input_ids, use_cache=False, return_dict=True)
        except TypeError:
            # Fallback for models that don't support use_cache/return_dict
            outputs = self.model(input_ids)
        
        # Extract logits from output
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, tuple):
            # If output is a tuple, first element is usually logits
            return outputs[0]
        else:
            # Direct tensor output
            return outputs


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
    
    Note:
        This converter uses a model wrapper that returns only logits to ensure
        compatibility with torch.export. Models like Qwen that return DynamicCache
        objects will work correctly with this approach. The wrapper sets use_cache=False
        and extracts only the logits tensor from the model output.
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
        
        # Wrap the model to return only logits (no cache)
        # This prevents issues with torch.export not supporting DynamicCache
        wrapped_model = ModelWrapper(model)
        
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
        # Pass only input_ids to avoid tracing issues with models like Qwen
        # that contain operations not supported by torch.fx during graph tracing
        # Use wrapped model to return only logits (no cache objects)
        edge_model = ai_edge_torch.convert(
            wrapped_model,
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
        
        # Determine if we're building a .litertlm file
        if build_litertlm and output_path.endswith('.litertlm'):
            # First save as TFLite
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use the desired output name as the base for the tflite file
                output_stem = Path(output_path).stem
                temp_tflite = Path(temp_dir) / f"{output_stem}.tflite"
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
                    fallback_path = str(Path(output_path).with_suffix('.tflite'))
                    edge_model.export(fallback_path)
                    print(f"✓ Conversion successful!")
                    print(f"  Model saved to: {fallback_path}")
                else:
                    # litertlm_builder creates the file as {tflite_stem}.litertlm in output_path directory
                    # So we build it in temp, then move to the desired location
                    litertlm_builder.build_litertlm(
                        tflite_model_path=str(temp_tflite),
                        workdir=temp_dir,
                        output_path=temp_dir,  # Build in temp directory first
                        context_length=max_seq_length,
                        hf_tokenizer_model_path=str(tokenizer_path),
                        llm_model_type='generic',
                    )
                    
                    # Move the generated .litertlm file to the desired output location
                    generated_file = Path(temp_dir) / f"{output_stem}.litertlm"
                    shutil.move(str(generated_file), output_path)
                    
                    print(f"✓ Conversion successful!")
                    print(f"  LiteRT LM model saved to: {output_path}")
        else:
            # Save as TFLite only
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
    
    # Determine output format and ensure proper extension
    if args.tflite_only:
        # User explicitly wants .tflite only
        build_litertlm_file = False
        if not args.output.endswith('.tflite'):
            args.output = str(Path(args.output).with_suffix('.tflite'))
    elif args.output.endswith('.tflite'):
        # Output ends with .tflite, don't build .litertlm
        build_litertlm_file = False
    else:
        # Default to building .litertlm
        build_litertlm_file = True
        if not args.output.endswith('.litertlm'):
            args.output = str(Path(args.output).with_suffix('.litertlm'))
    
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
