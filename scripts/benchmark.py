#!/usr/bin/env python3
"""
LLM Benchmark Tool - Automated Testing Framework
=================================================
A tool for benchmarking LLM models with structured question datasets.
Results are saved with proper naming conventions and rich metadata.
"""

import json
import subprocess
import time
import datetime
import os
import sys
import hashlib
import platform
import shutil

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Available datasets
DATASETS = {
    "easy": {
        "file": "questions_easy.json",
        "description": "Easy difficulty - Basic reasoning and knowledge",
        "difficulty": "easy"
    },
    "medium": {
        "file": "questions_medium.json", 
        "description": "Medium difficulty - Logic, math, and analysis",
        "difficulty": "medium"
    },
    "long_context": {
        "file": "questions_long_context_understanding.json",
        "description": "Long Context Understanding - Retrieval, tracking, reasoning",
        "difficulty": "hard"
    },
    "full": {
        "file": "questions.json",
        "description": "Full dataset - Comprehensive test suite",
        "difficulty": "mixed"
    }
}

# Terminal colors and symbols
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# Check marks and symbols
CHECK = "✓"
CROSS = "✗"
ARROW = "→"
ROCKET = "🚀"
FOLDER = "📂"
CHART = "📊"
DOC = "📄"
CLOCK = "⏱️"
BRAIN = "🧠"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_terminal_width():
    """Get terminal width for formatting."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def truncate_text(text, max_length=50):
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_response_preview(response, max_lines=3, max_width=70):
    """Format response for preview display."""
    lines = response.replace('\n', ' ').split()
    result = ' '.join(lines)
    if len(result) > max_width * max_lines:
        result = result[:max_width * max_lines - 3] + "..."
    return result


def get_installed_models():
    """Runs 'ollama list' to get models dynamically."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        models = []
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except Exception as e:
        print(f"{Colors.RED}Error fetching models: {e}{Colors.RESET}")
        return []


def get_test_history():
    """Load test history to track how many times each model was tested on each dataset."""
    history_file = os.path.join(RESULTS_DIR, ".test_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def update_test_history(model_name, dataset_key):
    """Update test history after a benchmark run."""
    history = get_test_history()
    key = f"{model_name}_{dataset_key}"
    history[key] = history.get(key, 0) + 1
    
    history_file = os.path.join(RESULTS_DIR, ".test_history.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    return history[key]


def get_model_test_count(model_name, dataset_key):
    """Get how many times a model has been tested on a dataset."""
    history = get_test_history()
    key = f"{model_name}_{dataset_key}"
    return history.get(key, 0)


def select_dataset():
    """Interactively asks user to pick a dataset."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          {BRAIN} SELECT QUESTION DATASET                          ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    datasets_list = list(DATASETS.items())
    
    for idx, (key, info) in enumerate(datasets_list, 1):
        file_path = os.path.join(DATA_DIR, info['file'])
        question_count = 0
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                    question_count = len(questions)
            except:
                question_count = 0
        
        difficulty_color = Colors.GREEN if info['difficulty'] == 'easy' else \
                          Colors.YELLOW if info['difficulty'] == 'medium' else Colors.MAGENTA
        
        status = f"{Colors.GREEN}[{question_count} questions]{Colors.RESET}" if question_count > 0 else f"{Colors.RED}[Not found]{Colors.RESET}"
        
        print(f"  {Colors.BOLD}{idx}.{Colors.RESET} {difficulty_color}{key.upper()}{Colors.RESET}")
        print(f"     {Colors.DIM}{info['description']}{Colors.RESET}")
        print(f"     {status}\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select a dataset (1-{len(datasets_list)}): {Colors.RESET}")
            choice = int(choice)
            if 1 <= choice <= len(datasets_list):
                selected_key = datasets_list[choice - 1][0]
                selected_info = datasets_list[choice - 1][1]
                file_path = os.path.join(DATA_DIR, selected_info['file'])
                
                if not os.path.exists(file_path):
                    print(f"{Colors.RED}Dataset file not found: {file_path}{Colors.RESET}")
                    continue
                    
                return selected_key, selected_info
            print(f"{Colors.RED}Invalid number. Please choose 1-{len(datasets_list)}.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")


def select_model(dataset_key):
    """Interactively asks user to pick a model, showing test counts."""
    models = get_installed_models()
    
    if not models:
        print(f"{Colors.RED}No models found via 'ollama list'. Is Ollama running?{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          {ROCKET} SELECT MODEL TO TEST                             ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    for idx, name in enumerate(models, 1):
        test_count = get_model_test_count(name, dataset_key)
        
        if test_count == 0:
            test_info = f"{Colors.DIM}(never tested on this dataset){Colors.RESET}"
        else:
            test_info = f"{Colors.YELLOW}(tested {test_count}x on this dataset){Colors.RESET}"
        
        print(f"  {Colors.BOLD}{idx}.{Colors.RESET} {Colors.GREEN}{name}{Colors.RESET} {test_info}")
    
    print()
    
    while True:
        try:
            choice = int(input(f"{Colors.CYAN}Select a model number to test: {Colors.RESET}"))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print(f"{Colors.RED}Invalid number.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a number.{Colors.RESET}")


def generate_run_id():
    """Generate a unique run ID based on timestamp and random hash."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{timestamp}_{random_suffix}"


def generate_result_filename(model_name, dataset_key, run_id):
    """Generate proper result filename with naming convention."""
    # Clean model name for filename
    clean_model = model_name.replace(':', '_').replace('/', '_')
    
    # Format: {dataset}_{model}_{run_id}.json (JSON only, no CSV)
    base_name = f"{dataset_key}_{clean_model}_{run_id}"
    
    return os.path.join(RESULTS_DIR, f"{base_name}.json")


def generate_question_id(item, index, dataset_key):
    """Generate improved question ID if not present."""
    if 'id' in item and item['id']:
        return item['id']
    
    # Generate new ID format: {DATASET}_{DOMAIN}_{INDEX:03d}
    domain_short = item.get('domain', 'GEN')[:3].upper()
    return f"{dataset_key.upper()}_{domain_short}_{index+1:03d}"


def get_system_info():
    """Collect system information for metadata."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "architecture": platform.machine()
    }


def load_questions(dataset_info):
    """Load questions from the selected dataset."""
    file_path = os.path.join(DATA_DIR, dataset_info['file'])
    
    if not os.path.exists(file_path):
        print(f"{Colors.RED}Error: {file_path} not found.{Colors.RESET}")
        sys.exit(1)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def display_progress_header(model_name, dataset_key, total_q, current_idx):
    """Display the progress header."""
    progress = (current_idx / total_q) * 100 if total_q > 0 else 0
    bar_width = 30
    filled = int(bar_width * current_idx / total_q) if total_q > 0 else 0
    bar = f"{'█' * filled}{'░' * (bar_width - filled)}"
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌{'─' * 68}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│{Colors.RESET} {BRAIN} Model: {Colors.GREEN}{model_name}{Colors.RESET} | Dataset: {Colors.YELLOW}{dataset_key.upper()}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│{Colors.RESET} Progress: [{bar}] {current_idx}/{total_q} ({progress:.1f}%)")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 68}┘{Colors.RESET}")


def display_recent_results(recent_results):
    """Display the 3 most recent prompts and responses."""
    if not recent_results:
        return
    
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔══ RECENT COMPLETIONS ══════════════════════════════════════════╗{Colors.RESET}")
    
    for i, result in enumerate(recent_results[-3:], 1):
        status_icon = f"{Colors.GREEN}{CHECK}{Colors.RESET}" if result['success'] else f"{Colors.RED}{CROSS}{Colors.RESET}"
        
        print(f"{Colors.MAGENTA}║{Colors.RESET}")
        print(f"{Colors.MAGENTA}║{Colors.RESET} {status_icon} [{result['index']}] {Colors.CYAN}{result['id']}{Colors.RESET} - {Colors.DIM}{result['test_for']}{Colors.RESET}")
        print(f"{Colors.MAGENTA}║{Colors.RESET}   {Colors.BOLD}Prompt:{Colors.RESET} {Colors.DIM}{truncate_text(result['prompt'], 55)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}║{Colors.RESET}   {Colors.BOLD}Response:{Colors.RESET} {Colors.GREEN}{truncate_text(result['response'], 55)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}║{Colors.RESET}   {CLOCK} {result['time']:.2f}s")
    
    print(f"{Colors.MAGENTA}╚{'═' * 68}╝{Colors.RESET}")


def run_benchmark():
    """Main benchmark execution function."""
    clear_screen()
    
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   {Colors.YELLOW}██╗     ██╗     ███╗   ███╗    ████████╗███████╗███████╗████████╗{Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ████╗ ████║    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝{Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ██╔████╔██║       ██║   █████╗  ███████╗   ██║   {Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ██║╚██╔╝██║       ██║   ██╔══╝  ╚════██║   ██║   {Colors.CYAN}   ║
║   {Colors.YELLOW}███████╗███████╗██║ ╚═╝ ██║       ██║   ███████╗███████║   ██║   {Colors.CYAN}   ║
║   {Colors.YELLOW}╚══════╝╚══════╝╚═╝     ╚═╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝   {Colors.CYAN}   ║
║                                                                   ║
║            Automated LLM Benchmark Framework v2.0                 ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")

    # 1. Select Dataset
    dataset_key, dataset_info = select_dataset()
    
    # 2. Select Model
    model_name = select_model(dataset_key)
    
    # 3. Generate unique run ID and filenames
    run_id = generate_run_id()
    result_file = generate_result_filename(model_name, dataset_key, run_id)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 4. Load questions
    questions = load_questions(dataset_info)
    total_q = len(questions)
    
    # Get test count for this run
    current_test_number = get_model_test_count(model_name, dataset_key) + 1
    
    clear_screen()
    print(f"\n{Colors.BOLD}{Colors.GREEN}{ROCKET} Starting Benchmark{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"  Model:        {Colors.CYAN}{model_name}{Colors.RESET}")
    print(f"  Dataset:      {Colors.YELLOW}{dataset_key.upper()}{Colors.RESET} ({total_q} questions)")
    print(f"  Test Number:  {Colors.MAGENTA}#{current_test_number}{Colors.RESET} for this model/dataset combo")
    print(f"  Run ID:       {Colors.DIM}{run_id}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")
    
    # Prepare metadata
    benchmark_start_time = datetime.datetime.now()
    metadata = {
        "run_id": run_id,
        "test_number": current_test_number,
        "model": model_name,
        "dataset": {
            "key": dataset_key,
            "file": dataset_info['file'],
            "difficulty": dataset_info['difficulty'],
            "total_questions": total_q
        },
        "execution": {
            "start_time": benchmark_start_time.isoformat(),
            "end_time": None,
            "total_duration_seconds": None
        },
        "system": get_system_info(),
        "statistics": {
            "total_questions": total_q,
            "successful": 0,
            "failed": 0,
            "average_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "total_response_time": 0
        }
    }
    
    # 5. Prepare results storage (JSON only)
    results_memory = []
    recent_display = []  # For displaying last 3 results

    # 6. Loop through questions
    for index, item in enumerate(questions):
        q_id = generate_question_id(item, index, dataset_key)
        prompt = item.get('prompt', '')
        
        # Clear and show progress
        clear_screen()
        display_progress_header(model_name, dataset_key, total_q, index)
        display_recent_results(recent_display)
        
        print(f"\n{Colors.YELLOW}{ARROW} Processing [{index+1}/{total_q}]: {Colors.CYAN}{q_id}{Colors.RESET}")
        print(f"  {Colors.DIM}Testing: {item.get('test_for', 'N/A')}{Colors.RESET}")
        print(f"  {Colors.DIM}Prompt: {truncate_text(prompt, 60)}{Colors.RESET}")
        
        start_time = time.time()
        timestamp = datetime.datetime.now().isoformat()
        
        response_text = ""
        success = True
        error_type = None
        
        try:
            # Run Ollama
            process = subprocess.run(
                ['ollama', 'run', model_name, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300  # 5 minute timeout
            )
            
            if process.returncode == 0:
                response_text = process.stdout.strip()
            else:
                response_text = f"ERROR: {process.stderr.strip()}"
                success = False
                error_type = "OLLAMA_ERROR"

        except subprocess.TimeoutExpired:
            response_text = "ERROR: Response timeout (5 minutes)"
            success = False
            error_type = "TIMEOUT"
        except Exception as e:
            response_text = f"SCRIPT ERROR: {str(e)}"
            success = False
            error_type = "SCRIPT_ERROR"
            
        end_time = time.time()
        duration = round(end_time - start_time, 3)
        
        # Calculate response metrics
        response_length_chars = len(response_text)
        response_length_words = len(response_text.split())
        
        # Update statistics
        metadata['statistics']['total_response_time'] += duration
        metadata['statistics']['min_response_time'] = min(metadata['statistics']['min_response_time'], duration)
        metadata['statistics']['max_response_time'] = max(metadata['statistics']['max_response_time'], duration)
        
        if success:
            metadata['statistics']['successful'] += 1
        else:
            metadata['statistics']['failed'] += 1
        
        status_icon = f"{Colors.GREEN}{CHECK}{Colors.RESET}" if success else f"{Colors.RED}{CROSS}{Colors.RESET}"
        print(f"  {status_icon} Completed in {duration:.2f}s ({response_length_words} words)")

        # 7. Add to memory for JSON dump
        result_entry = {
            "run_id": run_id,
            "test_number": current_test_number,
            "question_index": index + 1,
            "question_id": q_id,
            "domain": item.get('domain', ''),
            "category": item.get('category', ''),
            "difficulty": item.get('difficulty', dataset_info['difficulty']),
            "test_for": item.get('test_for', ''),
            "prompt": prompt,
            "model_tested": model_name,
            "timestamp": timestamp,
            "response": response_text,
            "metrics": {
                "response_time_seconds": duration,
                "response_length_chars": response_length_chars,
                "response_length_words": response_length_words
            },
            "execution_success": success,
            "error_type": error_type
        }
        results_memory.append(result_entry)
        
        # Add to recent display
        recent_display.append({
            'index': index + 1,
            'id': q_id,
            'test_for': item.get('test_for', 'N/A'),
            'prompt': prompt,
            'response': response_text,
            'time': duration,
            'success': success
        })
        
        # Keep only last 3
        if len(recent_display) > 3:
            recent_display.pop(0)

    # 9. Finalize metadata
    benchmark_end_time = datetime.datetime.now()
    total_duration = (benchmark_end_time - benchmark_start_time).total_seconds()
    
    metadata['execution']['end_time'] = benchmark_end_time.isoformat()
    metadata['execution']['total_duration_seconds'] = round(total_duration, 2)
    
    if total_q > 0:
        metadata['statistics']['average_response_time'] = round(
            metadata['statistics']['total_response_time'] / total_q, 3
        )
    
    if metadata['statistics']['min_response_time'] == float('inf'):
        metadata['statistics']['min_response_time'] = 0
    
    # Round min/max
    metadata['statistics']['min_response_time'] = round(metadata['statistics']['min_response_time'], 3)
    metadata['statistics']['max_response_time'] = round(metadata['statistics']['max_response_time'], 3)
    metadata['statistics']['total_response_time'] = round(metadata['statistics']['total_response_time'], 2)
    
    # Calculate success rate
    metadata['statistics']['success_rate'] = round(
        (metadata['statistics']['successful'] / total_q * 100) if total_q > 0 else 0, 2
    )
    
    # 10. Save final JSON with metadata
    final_json = {
        "metadata": metadata,
        "results": results_memory
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    # 11. Update test history
    update_test_history(model_name, dataset_key)
    
    # 12. Display final summary
    clear_screen()
    print(f"""
{Colors.BOLD}{Colors.GREEN}
╔═══════════════════════════════════════════════════════════════════╗
║                    {CHECK} BENCHMARK COMPLETE!                          ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")
    
    print(f"{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"  Model:              {Colors.CYAN}{model_name}{Colors.RESET}")
    print(f"  Dataset:            {Colors.YELLOW}{dataset_key.upper()}{Colors.RESET}")
    print(f"  Test Number:        {Colors.MAGENTA}#{current_test_number}{Colors.RESET}")
    print(f"  Run ID:             {Colors.DIM}{run_id}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"  Total Questions:    {total_q}")
    print(f"  Successful:         {Colors.GREEN}{metadata['statistics']['successful']}{Colors.RESET}")
    print(f"  Failed:             {Colors.RED}{metadata['statistics']['failed']}{Colors.RESET}")
    print(f"  Success Rate:       {metadata['statistics']['success_rate']}%")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"  Total Duration:     {total_duration:.2f}s")
    print(f"  Avg Response Time:  {metadata['statistics']['average_response_time']:.3f}s")
    print(f"  Min Response Time:  {metadata['statistics']['min_response_time']:.3f}s")
    print(f"  Max Response Time:  {metadata['statistics']['max_response_time']:.3f}s")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Results Saved:{Colors.RESET}")
    print(f"  {DOC} JSON: {Colors.GREEN}{result_file}{Colors.RESET}")
    
    print(f"\n{Colors.DIM}Thank you for using LLM Benchmark Tool!{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Benchmark interrupted by user.{Colors.RESET}")
        sys.exit(0)
