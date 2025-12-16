# LLM Benchmark - User Manual

A comprehensive tool for benchmarking LLM models with structured question datasets.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running Benchmarks](#running-benchmarks)
4. [Using Google Colab T4 GPU](#using-google-colab-t4-gpu)
5. [Backend Options](#backend-options)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)

---

## Quick Start

```bash
# Run with default Ollama backend
python scripts/benchmark.py

# Run with specific backend
python scripts/benchmark.py --backend transformers --use-4bit
```

---

## Installation

### Local (Ollama)

1. Install [Ollama](https://ollama.ai)
2. Pull models: `ollama pull llama3.2:3b`
3. Run benchmarks: `python scripts/benchmark.py`

### Transformers Backend (GPU)

```bash
pip install transformers torch accelerate bitsandbytes
python scripts/benchmark.py --backend transformers
```

---

## Running Benchmarks

### Basic Usage

```bash
python scripts/benchmark.py
```

This will:
1. Auto-detect the appropriate backend (Ollama locally, Transformers in Colab)
2. Show available datasets
3. Show available models
4. Run the benchmark interactively

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--backend` | Backend to use: `auto`, `ollama`, `transformers`, `remote` | `auto` |
| `--use-4bit` | Enable 4-bit quantization (saves VRAM) | `False` |
| `--device` | Device for Transformers: `cuda`, `cpu`, `auto` | `auto` |
| `--max-tokens` | Maximum tokens to generate | `512` |
| `--remote-url` | URL for remote Ollama server | `http://localhost:11434` |

### Examples

```bash
# Use Ollama explicitly
python scripts/benchmark.py --backend ollama

# Use Transformers with 4-bit quantization
python scripts/benchmark.py --backend transformers --use-4bit

# Connect to remote Ollama server
python scripts/benchmark.py --backend remote --remote-url http://server:11434

# Transformers with custom settings
python scripts/benchmark.py --backend transformers --max-tokens 1024 --device cuda
```

---

## Using Google Colab T4 GPU

Google Colab offers **free T4 GPUs** (16GB VRAM) that you can use to run benchmarks without needing a local GPU.

### Method 1: Use the Colab Notebook

1. Open `llm_benchmark_colab.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → Select **T4 GPU**
3. Run the cells in order:
   - Install dependencies
   - Mount Google Drive
   - Initialize backend
   - Run benchmarks
4. Results are saved to your Google Drive

### Method 2: Clone and Run

```python
# In Colab
!git clone https://github.com/your-username/llm-benchmark.git
%cd llm-benchmark

!pip install transformers torch accelerate bitsandbytes

# The benchmark will auto-detect Colab and use Transformers backend
!python scripts/benchmark.py
```

### T4 GPU Model Compatibility

| Model Size | FP16 | 4-bit Quantized |
|------------|------|-----------------|
| 1-3B params | ✅ Easy | ✅ Easy |
| 7B params | ⚠️ Tight | ✅ Good |
| 13B params | ❌ OOM | ⚠️ Possible |

### Recommended Models for T4

- `qwen2.5:1.5b` - Fast and capable
- `qwen2.5:3b` - Good balance
- `smollm2:1.7b` - Efficient
- `phi3.5:3.8b` - Strong reasoning
- `gemma2:2b` - Google's efficient model

### Tips for Colab

1. **Enable 4-bit**: Set `USE_4BIT = True` for larger models
2. **Clear cache**: Use `backend.clear_cache()` between models
3. **Save to Drive**: Mount Drive to persist results
4. **Session limits**: Free tier has usage limits, save frequently

---

## Backend Options

### OllamaBackend (Default for Local)

Uses local Ollama server via subprocess calls. 

**Pros:**
- Easy setup
- Wide model support
- Handles model management

**Requirements:**
- Ollama installed and running

### TransformersBackend (Default for Colab)

Uses HuggingFace Transformers with GPU acceleration.

**Pros:**
- Direct GPU control
- FP16 and 4-bit quantization
- No separate server needed

**Requirements:**
- `pip install transformers torch accelerate bitsandbytes`

### RemoteOllamaBackend

Connects to Ollama server via REST API.

**Use case:** Run Ollama on a remote server or via ngrok tunnel.

```bash
python scripts/benchmark.py --backend remote --remote-url http://your-server:11434
```

---

## Evaluation

Use Gemini AI to score model responses (0-10):

```bash
python scripts/evaluate.py
```

Follow the interactive prompts to select result files for evaluation.

---

## Visualization

Launch the interactive dashboard:

```bash
python scripts/visualize.py
```

Opens a Dash dashboard at `http://localhost:8050` with:
- Response time charts
- Success rate comparisons
- Per-question analysis
- Model comparisons

---

## Model Name Mappings

When using `TransformersBackend`, Ollama-style model names are mapped to HuggingFace:

| Ollama Name | HuggingFace Model |
|-------------|-------------------|
| `llama3.2:1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama3.2:3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen2.5:3b` | `Qwen/Qwen2.5-3B-Instruct` |
| `gemma2:2b` | `google/gemma-2-2b-it` |
| `phi3.5:3.8b` | `microsoft/Phi-3.5-mini-instruct` |
| `smollm2:1.7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

You can also use HuggingFace model paths directly (e.g., `microsoft/Phi-3.5-mini-instruct`).

---

## Troubleshooting

### "No models found via 'ollama list'"
- Ensure Ollama is installed: `which ollama`
- Start Ollama server: `ollama serve`
- Pull at least one model: `ollama pull llama3.2:3b`

### CUDA out of memory
- Use 4-bit quantization: `--use-4bit`
- Use a smaller model
- Clear cache: `backend.clear_cache()` in Python

### Gated model access denied (Llama)
- Accept license at huggingface.co/meta-llama
- Login: `huggingface-cli login`

### Slow response times
- Check GPU usage: `nvidia-smi`
- Reduce `--max-tokens`
- Use FP16 instead of FP32
