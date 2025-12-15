# 🧠 LLM Benchmark Pipeline - Complete User Guide

> **For beginners**: This guide assumes you're new to development. Every step is explained in detail with copy-paste commands. Look for the `#hashtag` comments - they're prompts you can copy to an AI chatbot for help customizing the setup.

---

## 📋 Table of Contents

1. [What This Tool Does](#what-this-tool-does)
2. [Prerequisites](#prerequisites)
3. [Setup (First Time Only)](#setup-first-time-only)
4. [Pipeline Overview](#pipeline-overview)
5. [Step 1: Run Benchmark](#step-1-run-benchmark)
6. [Step 2: Evaluate Results](#step-2-evaluate-results)
7. [Step 3: Visualize Data](#step-3-visualize-data)
8. [File Structure](#file-structure)
9. [Scoring System](#scoring-system)
10. [Troubleshooting](#troubleshooting)
11. [Customization Prompts](#customization-prompts-for-ai-assistants)

---

## What This Tool Does

This pipeline has 3 stages:

| Stage | What It Does | Input | Output |
|-------|--------------|-------|--------|
| **Benchmark** | Sends questions to an LLM and records responses | Question dataset | `results/*.json` |
| **Evaluate** | Uses Gemini AI to score each response (0-10) | Results JSON | `evaluations/{model}/{dataset}/*.json` |
| **Visualize** | Shows interactive charts in browser | Evaluation JSON | Web dashboard |

**Why these 3 stages?**
1. **Benchmark** tests how your LLM responds to different questions
2. **Evaluate** uses a more powerful AI (Gemini) to grade those responses objectively
3. **Visualize** helps you understand patterns - which question types the model handles well/poorly

---

## Prerequisites

### Required Software

1. **Python 3.10+** - The programming language
2. **Ollama** - Runs local LLM models on your computer
3. **Gemini API Key** - Free key from Google for evaluation

### Check Python Version

```bash
python --version
```

# ℹ️ You should see Python 3.10 or higher
# #python_install If you see "command not found" or a version below 3.10, ask: "How do I install Python 3.10+ on [your OS: Linux/Mac/Windows]?"

### Check/Install Ollama

```bash
ollama --version
```

# ℹ️ If not installed, visit: https://ollama.ai
# #ollama_install Ask: "How do I install Ollama on [your OS]?"

### Start Ollama Server

```bash
ollama serve
```

# ⚠️ WARNING: Keep this running in a separate terminal!
# Ollama must be running for benchmarks to work

### Download a Model for Testing

```bash
ollama pull gemma3:270m
```

# ℹ️ This downloads a small 270MB model for testing
# Other small models: smollm2:135m, granite4:350m
# #model_choice Ask: "What Ollama models are good for [your use case]?"

### Get Gemini API Key

1. Go to: https://aistudio.google.com/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

# 🔐 SECURITY: Never share your API key publicly!

---

## Setup (First Time Only)

### Step 1: Navigate to Project Folder

```bash
cd /home/adi/PROGRAMMING/LLM-TEST
```

# 🔧 CUSTOMIZATION NEEDED: Change this path to YOUR project location!
# #path_linux Linux users: Use path like /home/username/LLM-TEST
# #path_mac Mac users: Use path like /Users/username/LLM-TEST  
# #path_windows Windows users: Use path like C:\Users\username\LLM-TEST
# #path_change Ask: "Convert this Linux path to Windows: /home/adi/PROGRAMMING/LLM-TEST"

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

# ℹ️ WHAT IS THIS?
# A virtual environment is an isolated Python installation
# It keeps this project's packages separate from your system Python
# The .venv folder will be created in your project directory

### Step 3: Activate Virtual Environment

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

# ✅ SUCCESS CHECK: Your terminal prompt should now show (.venv) at the start
# Example: (.venv) adi@computer:~/LLM-TEST$
# 
# ⚠️ WARNING: You MUST activate the venv every time you open a new terminal!
# #venv_activate Ask: "How do I activate a Python virtual environment on [your OS]?"

### Step 4: Install Dependencies

```bash
pip install dash dash-bootstrap-components plotly pandas google-generativeai
```

# ⏳ This may take 1-2 minutes to download all packages
# ✅ You should see "Successfully installed..." at the end
# 
# ⚠️ If you see "pip not found": Make sure venv is activated!
# #pip_error Ask: "pip command not found after activating venv on [your OS]"

### Step 5: Set Up Gemini API Key

**Option A - Environment Variable (temporary, resets when terminal closes):**

Linux/Mac:
```bash
export GEMINI_API_KEY='AIzaSy...'
```

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY='AIzaSy...'
```

Windows CMD:
```cmd
set GEMINI_API_KEY=AIzaSy...
```

**Option B - Create .env File (permanent, recommended):**

```bash
echo "GEMINI_API_KEY=AIzaSy..." > .env
```

# 🔧 Replace AIzaSy... with YOUR actual API key!
# 🔐 SECURITY: Add .env to .gitignore so you don't accidentally commit it
# #api_key_help Ask: "How do I securely store API keys for a Python project?"

### Step 6: Verify Setup

```bash
# Check packages installed
pip list | grep -E "dash|plotly|google"

# Check Ollama models
ollama list
```

# ✅ You should see dash, plotly, google-generativeai in pip list
# ✅ You should see at least one model in ollama list

---

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   BENCHMARK     │     │    EVALUATE     │     │   VISUALIZE     │
│   benchmark.py  │ ──► │   evaluate.py   │ ──► │  visualize.py   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   results/              evaluations/               Browser
   *.json             {model}/{dataset}/        localhost:8050
                        *_eval.json
```

---

## Step 1: Run Benchmark

### What This Does
- Sends each question from a dataset to your chosen LLM model
- Records the model's response and how long it took
- Saves everything to a JSON file (no CSV files!)

### Command

```bash
# Make sure you're in the project folder and venv is activated!
cd /home/adi/PROGRAMMING/LLM-TEST
source .venv/bin/activate

# Run the benchmark
.venv/bin/python scripts/benchmark.py
```

# 🔧 CUSTOMIZATION: Change the cd path to YOUR project location
# #path_change Ask: "Convert this path for Windows: /home/adi/PROGRAMMING/LLM-TEST"
#
# ⚠️ We use .venv/bin/python to ensure we're using the venv Python
# On Windows: .venv\Scripts\python scripts\benchmark.py

### Interactive Prompts

1. **Select Dataset**: 
   - Choose `1` for EASY (30 questions) - good for testing
   - MEDIUM and LONG_CONTEXT are harder

2. **Select Model**:
   - Pick a number from the list of your installed Ollama models

### Example Run

```
╔══════════════════════════════════════════════════════════════╗
║          🧠 SELECT QUESTION DATASET                          ║
╚══════════════════════════════════════════════════════════════╝

  1. EASY
     Easy difficulty - Basic reasoning and knowledge
     [30 questions]

Select a dataset (1-4): 1

╔══════════════════════════════════════════════════════════════╗
║          🚀 SELECT MODEL TO TEST                             ║
╚══════════════════════════════════════════════════════════════╝

  1. smollm2:135m
  2. gemma3:270m

Select a model: 2
```

### Output

```
✓ BENCHMARK COMPLETE
════════════════════════════════════════════════════════════
  Total: 30 | Success: 30 | Failed: 0
  Avg Time: 0.276s

  📄 Saved: results/easy_gemma3_270m_20251215_144447_3636bc.json
```

# ⏳ TIMING: Easy dataset with small model takes ~30 seconds to 2 minutes
# 📁 OUTPUT: JSON file saved in results/ folder

---

## Step 2: Evaluate Results

### What This Does
- Reads the benchmark results JSON you just created
- Sends each response to Google's Gemini AI for strict scoring (0-10)
- Saves evaluation with scores and remarks in organized folders

### Command

```bash
.venv/bin/python scripts/evaluate.py
```

# ⚠️ REQUIRES: Gemini API key must be set (see Setup Step 5)
# On Windows: .venv\Scripts\python scripts\evaluate.py

### Interactive Prompts

1. **Select Result File**: Choose the JSON file you just created in Step 1
2. **API Key**: Enter your Gemini API key if not already set

### Example Run

```
╔════════════════════════════════════════════╗
║  🔍 LLM EVALUATOR (Strict 0-10 Scoring)    ║
║     Powered by Gemini                      ║
╚════════════════════════════════════════════╝

Available result files:

  1. easy_gemma3_270m_20251215_144447_3636bc.json
     gemma3:270m | easy | 30 questions

Select file (1-1): 1

Loading: easy_gemma3_270m_20251215_144447_3636bc.json
Found 30 responses to evaluate

[1/30] EASY_LOGIC_001 2/10 - Completely incorrect conclusion...
[2/30] EASY_LOGIC_002 8/10 - Correct answer with clear reasoning...
...

════════════════════════════════════════════════════════
✓ EVALUATION COMPLETE
════════════════════════════════════════════════════════
  Evaluated:     30/30
  Average Score: 5.2/10

  Saved: evaluations/gemma3_270m/easy/gemma3_270m_easy_20251215_144500_eval.json
```

# ⏳ TIMING: ~1-2 minutes for 30 questions (Gemini has rate limits)
# 💰 COST: Gemini API free tier has generous limits (enough for testing)
# 📁 OUTPUT: JSON saved in evaluations/{model}/{dataset}/ folder

### Output Folder Structure

```
evaluations/
└── gemma3_270m/           <- Model name
    └── easy/              <- Dataset name
        └── gemma3_270m_easy_20251215_144500_eval.json
```

---

## Step 3: Visualize Data

### What This Does
- Starts a web dashboard on your computer
- Upload evaluation JSON files to see interactive charts
- Compare multiple models side-by-side

### Command

```bash
.venv/bin/python scripts/visualize.py
```

# On Windows: .venv\Scripts\python scripts\visualize.py

### Output

```
╔═══════════════════════════════════════════════════════════════════╗
║   🧠 LLM BENCHMARK VISUALIZATION DASHBOARD                        ║
╚═══════════════════════════════════════════════════════════════════╝

  Starting server on: http://localhost:8050
  
  Press Ctrl+C to stop the server.
```

### Open in Browser

1. Open your web browser (Chrome, Firefox, etc.)
2. Go to: **http://localhost:8050**
3. You'll see an upload area

### Upload Evaluation Files

1. Navigate to: `evaluations/{model}/{dataset}/`
2. Drag & drop the `*_eval.json` file onto the upload area
3. Charts will appear!

# 🌐 The dashboard runs on YOUR computer (localhost means "this computer")
# 📂 Upload files from: evaluations/ folder (NOT results/ folder!)
# ⚠️ Common mistake: Uploading raw results/*.json instead of evaluations/*_eval.json

### Available Charts

| Chart | X-Axis | Y-Axis | What It Shows |
|-------|--------|--------|---------------|
| **Score vs Response Time** | Score (0-10) | Time (seconds) | Do slower responses score better? |
| **Category Comparison** | Category | Average Score | Which question types are easiest/hardest |
| **Score Distribution** | Score | Count | Histogram of all scores |
| **Model Comparison** | Model | Average Score | Compare multiple models |

### Filters

- **Score Range Slider**: Filter to show only certain score ranges
- **Quick Filters**:
  - `Low (<3)`: Show only poor responses
  - `Mid (3-7)`: Show average responses
  - `High (8+)`: Show good responses
  - `Perfect (10)`: Show only perfect scores
  - `All`: Reset to show everything

### Compare Multiple Models

1. Run benchmarks with different models (Step 1)
2. Evaluate each benchmark (Step 2)
3. Upload multiple evaluation files at once
4. See comparison charts!

### Stop the Dashboard

Press `Ctrl+C` in the terminal to stop the server.

---

## File Structure

```
LLM-TEST/
├── .venv/                    # Virtual environment (created by setup)
│                             # ⚠️ Don't modify this folder!
│
├── .env                      # Your API key (YOU create this)
│                             # 🔐 Don't commit to git!
│
├── data/                     # Question datasets (don't modify)
│   ├── questions_easy.json       # 30 easy questions
│   ├── questions_medium.json     # 30 medium questions
│   └── questions_long_context_understanding.json
│
├── results/                  # Benchmark outputs (auto-generated)
│   └── {dataset}_{model}_{timestamp}.json
│       # Contains: prompts, responses, response times
│
├── evaluations/              # Evaluation outputs (auto-generated)
│   └── {model}/
│       └── {dataset}/
│           └── {model}_{dataset}_{timestamp}_eval.json
│               # Contains: scores, remarks from Gemini
│
├── scripts/
│   ├── benchmark.py          # Step 1: Test LLM
│   ├── evaluate.py           # Step 2: Score responses
│   └── visualize.py          # Step 3: View charts
│
└── docs/
    └── USER_MANUAL.md        # This file!
```

---

## Scoring System

### Strict 0-10 Scale

The evaluation uses a **strict** scoring system. Most responses score 4-7.

| Score | Rating | Meaning | Color |
|-------|--------|---------|-------|
| 0-2 | ❌ Fail | Completely wrong, irrelevant, or harmful | Red |
| 3-4 | ⚠️ Poor | Major errors or significant omissions | Orange |
| 5-6 | 📊 Mediocre | Acceptable but has minor errors | Yellow |
| 7-8 | ✅ Good | Mostly correct and complete | Light Green |
| 9 | ⭐ Excellent | Near-perfect response | Green |
| 10 | 🏆 Perfect | Flawless (extremely rare!) | Bright Green |

### Why Strict Scoring?

- Easy to see which models actually perform well
- Prevents "everyone gets an A" problem
- Score of 7+ means the response is genuinely good

### Evaluation JSON Format

Each evaluation entry looks like:

```json
{
  "question_id": "EASY_LOGIC_001",
  "prompt": "If all cats are animals, and Whiskers is a cat...",
  "response": "Whiskers is an animal.",
  "response_time": 0.35,
  "domain": "Logic",
  "category": "Basic Reasoning",
  "score": 9,
  "remark": "Correct logical deduction with clear answer."
}
```

---

## Troubleshooting

### "google-generativeai package not installed"

**Cause**: Running with system Python instead of virtual environment

**Fix**:
```bash
# Activate virtual environment first!
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Then run the script
.venv/bin/python scripts/evaluate.py
```

# ⚠️ REMEMBER: Always activate venv before running scripts!

### "No models found via 'ollama list'"

**Cause**: Ollama is not running or no models installed

**Fix**:
```bash
# In a separate terminal, start Ollama:
ollama serve

# Pull a model:
ollama pull gemma3:270m
```

# ⚠️ Keep "ollama serve" running in background!

### "GEMINI_API_KEY not found"

**Cause**: API key not set

**Fix**:
```bash
# Option 1: Set environment variable
export GEMINI_API_KEY='your-key-here'

# Option 2: Create .env file
echo "GEMINI_API_KEY=your-key-here" > .env
```

# 🔧 Replace 'your-key-here' with your actual Gemini API key

### "Port 8050 already in use"

**Cause**: Another process is using port 8050

**Fix (Linux/Mac)**:
```bash
# Find process using port 8050
lsof -i :8050

# Kill it (replace 12345 with actual PID)
kill -9 12345
```

**Fix (Windows)**:
```cmd
netstat -ano | findstr :8050
taskkill /PID 12345 /F
```

# #port_change Ask: "How do I change the port for a Python Dash app from 8050 to 8080?"

### Dashboard shows "No data matching filters"

**Cause**: Uploaded wrong file type

**Fix**: Upload files from `evaluations/` folder, NOT `results/` folder!
- ❌ Wrong: `results/easy_gemma3_*.json`
- ✅ Correct: `evaluations/gemma3_270m/easy/*_eval.json`

### Evaluation shows all "error" for every question

**Cause**: Invalid or empty Gemini API key

**Fix**: Make sure your API key is correct:
1. Go to https://aistudio.google.com/apikey
2. Copy the full key (starts with `AIza...`)
3. Set it in your environment or .env file

---

## Customization Prompts (for AI Assistants)

Use these prompts with ChatGPT, Claude, or other AI assistants to customize this tool:

### Change Project Path

```
#path_change 
I have a Python project that uses paths like /home/adi/PROGRAMMING/LLM-TEST
I need to convert all paths for [Windows 11 / Mac / different Linux user].
My username is [YOUR_USERNAME].
Also update any bash commands (source .venv/bin/activate) to work on my OS.
```

### Change Dashboard Port

```
#port_change
In my Python Dash app (visualize.py), the server runs on port 8050.
I want to change it to port [NEW_PORT].
Show me what line to modify and what to change it to.
```

### Add New Question Dataset

```
#add_dataset
I have a benchmark tool with datasets defined in benchmark.py like this:
DATASETS = {
    "easy": {"file": "questions_easy.json", "difficulty": "easy"},
    ...
}

I want to add a new dataset called [NAME] with [DIFFICULTY] difficulty.
The file is at data/[FILENAME].json.
Show me the code to add.
```

### Change Scoring Scale

```
#scoring_change
My evaluation script (evaluate.py) uses a 0-10 scoring scale.
I want to change it to [0-100 / 0-5 / letter grades].
What parts of the code need to change?
Also update the prompt sent to Gemini and the visualization.
```

### Use Different Gemini Model

```
#model_change
My evaluate.py uses gemini-2.0-flash for evaluation.
I want to use [gemini-1.5-pro / gemini-1.5-flash] instead.
What line do I change?
```

### Windows-Specific Setup

```
#windows_setup
I have setup instructions written for Linux/Mac.
Convert all commands to Windows PowerShell:
- source .venv/bin/activate
- .venv/bin/python scripts/benchmark.py
- export GEMINI_API_KEY='...'
- echo "..." > .env
```

### Add New Chart Type

```
#add_chart
I have a Dash visualization (visualize.py) with charts for:
- Score vs Response Time (scatter)
- Category Comparison (bar)
- Score Distribution (histogram)
- Model Comparison (bar)

I want to add a new chart showing [DESCRIPTION].
Show me the Plotly/Dash code to add this chart.
```

### Filter by Domain

```
#add_filter
My visualization dashboard has filters for score ranges.
I want to add a dropdown filter to filter by domain (Logic, Math, Language, etc.).
Show me how to add this to my Dash callbacks.
```

---

## Quick Reference Commands

```bash
# 1. Always start with these!
cd /home/adi/PROGRAMMING/LLM-TEST
source .venv/bin/activate

# 2. Run the pipeline
.venv/bin/python scripts/benchmark.py      # Step 1: Test LLM
.venv/bin/python scripts/evaluate.py       # Step 2: Score with Gemini
.venv/bin/python scripts/visualize.py      # Step 3: View dashboard

# 3. Open browser
# Go to: http://localhost:8050
```

# 🔧 Remember to change the path for your system!
# #quickstart Ask: "Give me the quick start commands for [your OS]"

---

## Demo Run (Copy-Paste Tutorial)

### Step-by-step for first-time users:

```bash
# 1. Open terminal and navigate to project
cd /home/adi/PROGRAMMING/LLM-TEST

# 2. Activate virtual environment
source .venv/bin/activate
# You should see (.venv) in your prompt

# 3. Make sure Ollama is running (in a separate terminal)
# ollama serve

# 4. Run benchmark
# When prompted: Select 1 for easy dataset, then select your model
.venv/bin/python scripts/benchmark.py

# 5. Run evaluation  
# When prompted: Select the file you just created
# Enter your Gemini API key if asked
.venv/bin/python scripts/evaluate.py

# 6. Start visualization
.venv/bin/python scripts/visualize.py

# 7. Open browser and go to:
# http://localhost:8050

# 8. Upload the evaluation file from:
# evaluations/{model}/{dataset}/*_eval.json
```

---

*Last updated: December 15, 2025*

---

## Need Help?

If you're stuck, copy this prompt to an AI assistant:

```
I'm using the LLM Benchmark tool and I'm stuck at [describe your problem].
I'm on [Windows/Mac/Linux].
The error message is: [paste error]
What should I do?
```
