# llm2litertlm

Convert Hugging Face LLM models to LiteRT (TensorFlow Lite) format for deployment on mobile and edge devices.

## Overview

This tool provides an easy way to convert Large Language Models (LLMs) from Hugging Face to LiteRT format using [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch). LiteRT (formerly TensorFlow Lite) enables running ML models efficiently on mobile, embedded, and edge devices.

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

### Basic Conversion

Convert a Hugging Face model to LiteRT format:

```bash
python convert.py --model gpt2 --output models/gpt2.tflite
```

### With Quantization

Apply quantization to reduce model size (with potential minor accuracy trade-off):

```bash
python convert.py --model gpt2 --output models/gpt2.tflite --quantize
```

### Custom Sequence Length

Specify a custom maximum sequence length:

```bash
python convert.py --model gpt2 --output models/gpt2.tflite --max-seq-length 1024
```

## Command Line Arguments

- `--model`, `-m`: (Required) Name or path of the Hugging Face model
  - Examples: `gpt2`, `facebook/opt-125m`, `EleutherAI/gpt-neo-125m`
- `--output`, `-o`: (Required) Output path for the converted LiteRT model
  - Example: `models/my_model.tflite`
- `--quantize`, `-q`: (Optional) Enable quantization to reduce model size
- `--max-seq-length`: (Optional) Maximum sequence length (default: 512)

## Examples

### Convert GPT-2

```bash
python convert.py --model gpt2 --output models/gpt2.tflite
```

### Convert OPT-125M with Quantization

```bash
python convert.py --model facebook/opt-125m --output models/opt-125m.tflite --quantize
```

### Convert a Local Model

```bash
python convert.py --model ./my_local_model --output models/custom.tflite
```

## Output

The conversion process will create:
- The converted LiteRT model (`.tflite` file)
- A tokenizer directory with the model's tokenizer configuration

## Supported Models

This tool supports most causal language models available on Hugging Face, including:
- GPT-2 and variants
- GPT-Neo
- OPT
- Llama (with appropriate access)
- And many more PyTorch-based LLMs

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