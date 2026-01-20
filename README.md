# llm2litertlm

Convert Hugging Face LLM models to LiteRT LM (.litertlm) format for deployment on mobile and edge devices.

## Overview

This tool provides an easy way to convert Large Language Models (LLMs) from Hugging Face to LiteRT LM format using [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch). The `.litertlm` format is a packaged format that includes the TFLite model, tokenizer, and metadata, enabling efficient deployment on mobile, embedded, and edge devices.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/wannaphong/llm2litertlm.git
cd llm2litertlm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8 or higher
- PyTorch 2.6+
- Transformers 4.48+
- ai-edge-torch 0.2.0+

## Usage

### Auto-Detect Converter (New - Recommended for Gemma, Llama, Phi, Qwen)

The new auto-detect converter automatically detects model architecture and uses optimized model builders:

```bash
python convert_autodetect.py
```

Then enter the Hugging Face Model ID when prompted (e.g., `google/gemma-2b-it`).

**Features:**
- Auto-installs dependencies if needed
- Auto-detects architecture from config.json
- Uses architecture-specific model builders (Gemma, Llama, Phi, Qwen)
- Int8 quantization for optimized performance
- Creates .litertlm bundle with MediaPipe

**Supported Architectures:**
- Gemma & Gemma2
- Llama (including Llama 2 and 3)
- Phi (Phi-2)
- Qwen2

### Basic Conversion (Generic Approach)

Convert a Hugging Face model to LiteRT LM (.litertlm) format using the generic converter:

```bash
python convert.py --model gpt2 --output models/gpt2.litertlm
```

The `.litertlm` format includes:
- The converted TFLite model
- Tokenizer configuration
- Model metadata (context length, etc.)

### TFLite Only

Convert to TFLite format only (without packaging):

```bash
python convert.py --model gpt2 --output models/gpt2.tflite --tflite-only
```

### With Quantization

Apply quantization to reduce model size (with potential minor accuracy trade-off):

```bash
python convert.py --model gpt2 --output models/gpt2.litertlm --quantize
```

### Custom Sequence Length

Specify a custom maximum sequence length:

```bash
python convert.py --model gpt2 --output models/gpt2.litertlm --max-seq-length 1024
```

## Command Line Arguments

- `--model`, `-m`: (Required) Name or path of the Hugging Face model
  - Examples: `gpt2`, `facebook/opt-125m`, `EleutherAI/gpt-neo-125m`
- `--output`, `-o`: (Required) Output path for the converted model
  - Examples: `models/my_model.litertlm`, `models/my_model.tflite`
- `--quantize`, `-q`: (Optional) Enable quantization to reduce model size
- `--max-seq-length`: (Optional) Maximum sequence length (default: 512, range: 1-8192)
- `--tflite-only`: (Optional) Only generate TFLite file without building .litertlm package

## Examples

### Convert GPT-2 to LiteRT LM

```bash
python convert.py --model gpt2 --output models/gpt2.litertlm
```

### Convert OPT-125M with Quantization

```bash
python convert.py --model facebook/opt-125m --output models/opt-125m.litertlm --quantize
```

### Convert to TFLite Only

```bash
python convert.py --model gpt2 --output models/gpt2.tflite --tflite-only
```

### Convert a Local Model

```bash
python convert.py --model ./my_local_model --output models/custom.litertlm
```

## Output

The conversion process will create:

### For .litertlm format (default):
- A complete `.litertlm` package containing:
  - The converted TFLite model
  - Tokenizer configuration
  - Model metadata (context length, prompt templates, etc.)

### For .tflite format (with --tflite-only):
- The converted `.tflite` model file
- A separate tokenizer directory with the model's tokenizer configuration

## Supported Models

### Auto-Detect Converter
Best for these specific architectures with optimized performance:
- **Gemma & Gemma2** (e.g., `google/gemma-2b-it`)
- **Llama** including Llama 2 and 3 (e.g., `meta-llama/Llama-2-7b-chat-hf`)
- **Phi** (e.g., `microsoft/phi-2`)
- **Qwen2** (e.g., `Qwen/Qwen2-0.5B`)

The auto-detect converter uses architecture-specific model builders from ai-edge-torch for better performance and Int8 quantization.

### Generic Converter (convert.py)
This tool supports most causal language models available on Hugging Face, including:
- GPT-2 and variants
- GPT-Neo
- OPT
- And many more PyTorch-based LLMs

Use the generic converter for models not explicitly supported by the auto-detect converter.

**Note**: Some models may require special handling or may not be fully supported by ai-edge-torch. Check the [ai-edge-torch documentation](https://github.com/google-ai-edge/ai-edge-torch) for the latest compatibility information.

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
Large models require significant memory. Consider:
- Using a machine with more RAM
- Using smaller model variants
- Enabling quantization with `--quantize`

### Model Compatibility
Some model architectures may not be fully supported. Check:
- ai-edge-torch documentation for supported models
- Model compatibility with PyTorch export

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## References

- [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch) - Google's library for converting PyTorch models to TensorFlow Lite
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - State-of-the-art ML models
- [LiteRT (TensorFlow Lite)](https://www.tensorflow.org/lite) - Deploy models on mobile and edge devices