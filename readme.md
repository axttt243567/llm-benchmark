# 🧠 LLM Benchmark Tool

A comprehensive benchmarking and evaluation framework for testing LLM models using Ollama, with Gemini-powered evaluation and interactive visualization dashboard.

## 📁 Project Structure

```
LLM-TEST/
├── data/                          # Question datasets
│   ├── questions_easy.json        # 30 easy questions
│   ├── questions_medium.json      # 30 medium questions
│   ├── questions_long_context_understanding.json
│   └── questions.json             # Full dataset (300 questions)
├── scripts/
│   ├── benchmark.py               # Main benchmark runner
│   ├── evaluate.py                # Gemini-powered evaluation
│   └── serve_dashboard.py         # Dashboard web server
├── results/                       # Benchmark outputs (CSV + JSON)
├── evaluations/                   # Gemini evaluation results
├── dashboard/
│   └── index.html                 # Interactive visualization UI
└── .env                           # API keys (create from .env.example)
```

## 🚀 Quick Start

### 1. Run Benchmark

```bash
cd /home/adi/PROGRAMMING/LLM-TEST/scripts
python benchmark.py
```

- Select a dataset (Easy, Medium, Long Context, or Full)
- Select a model from your Ollama installation
- Watch real-time progress with the last 3 responses displayed
- Results saved with unique naming: `{dataset}_{model}_{timestamp}.json`

### 2. Evaluate Results with Gemini

```bash
# First, set up your API key
cp .env.example .env
# Edit .env and add your Gemini API key

# Run evaluation
python evaluate.py
```

Or with command line arguments:
```bash
python evaluate.py --file ../results/your_result_file.json --api-key YOUR_KEY
```

### 3. View Dashboard

```bash
python serve_dashboard.py
```

Opens at `http://localhost:8080` with:
- 📊 Grade distribution charts
- 🎯 Score dimension radar chart
- 📈 Domain/category performance
- 🗺️ Performance heatmap
- ⚖️ Multi-model comparison

## 📊 Output Formats

### Benchmark Results JSON
```json
{
  "metadata": {
    "run_id": "20251215_122800_abc123",
    "model": "llama2:7b",
    "dataset": {"key": "easy", "difficulty": "easy"},
    "statistics": {"pass_rate": 93.33, "avg_response_time": 0.8}
  },
  "results": [...]
}
```

### Evaluation Results JSON
```json
{
  "evaluation_id": "eval_20251215_130000_xyz789",
  "evaluator_model": "gemini-2.5-pro",
  "summary": {
    "pass_rate": 85.0,
    "grade_distribution": {"A": 10, "B": 8, "C": 5, "D": 2, "F": 5},
    "average_scores": {
      "correctness": 78,
      "completeness": 82,
      "clarity": 85,
      "reasoning": 75,
      "relevance": 88,
      "overall": 80
    }
  },
  "evaluations": [...]
}
```

## 🔧 Requirements

- Python 3.8+
- Ollama (running locally)
- Google Generative AI package: `pip install google-generativeai`
- Gemini API key (for evaluation)

## 📝 Adding New Datasets

Add your dataset to the `DATASETS` dictionary in `scripts/benchmark.py`:

```python
DATASETS = {
    "your_dataset": {
        "file": "questions_your_dataset.json",
        "description": "Your description here",
        "difficulty": "easy/medium/hard/mixed"
    },
    # ...
}
```

## 📄 License

MIT
