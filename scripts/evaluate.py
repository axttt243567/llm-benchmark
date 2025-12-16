#!/usr/bin/env python3
"""
LLM Evaluation Tool - Batch Evaluation with Gemini AI
======================================================
Evaluates entire benchmark result files using Gemini AI with strict 0-10 scoring.
Supports batch processing of multiple files automatically.
Stores results in: evaluations/{llm_model}/{dataset}/
"""

import json
import os
import sys
import datetime
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import google.generativeai as genai

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
DATA_DIR = PROJECT_ROOT / "data"

# Available Gemini models for evaluation (2025 latest models)
GEMINI_MODELS = {
    "1": {"name": "gemini-2.5-pro", "description": "Most capable, best quality (slower)"},
    "2": {"name": "gemini-2.5-flash", "description": "Fast & smart, great balance (recommended)"},
    "3": {"name": "gemini-2.5-flash-lite", "description": "Ultra-fast, lightweight evaluation"},
    "4": {"name": "gemini-2.0-flash", "description": "Previous gen, stable & reliable"},
}

# Available datasets - organized in category folders
DATASETS = {
    "1": {"key": "long_context_retrieval", "file": "long_context/retrieval.json", "description": "Long Context - Information retrieval"},
    "2": {"key": "long_context_narrative", "file": "long_context/narrative.json", "description": "Long Context - Narrative comprehension"},
    "3": {"key": "tricky_logical", "file": "tricky_questions/logical_traps.json", "description": "Tricky - Logical traps"},
    "4": {"key": "tricky_ambiguity", "file": "tricky_questions/ambiguity.json", "description": "Tricky - Linguistic ambiguity"},
    "5": {"key": "math_algebra", "file": "math_10th_grade/algebra.json", "description": "Math 10th Grade - Algebra"},
    "6": {"key": "math_geometry", "file": "math_10th_grade/geometry.json", "description": "Math 10th Grade - Geometry"},
    "7": {"key": "common_sense_everyday", "file": "common_sense/everyday.json", "description": "Common Sense - Everyday"},
    "8": {"key": "common_sense_social", "file": "common_sense/social.json", "description": "Common Sense - Social"},
    "9": {"key": "persona_characters", "file": "persona_roleplay/characters.json", "description": "Persona - Characters"},
    "10": {"key": "persona_scenarios", "file": "persona_roleplay/scenarios.json", "description": "Persona - Scenarios"},
}

# Terminal colors
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


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_api_key():
    """Get Gemini API key from environment or .env file."""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.strip().split('=', 1)[1].strip().strip('"\'')
                        break
    
    return api_key


def get_all_result_files():
    """Get all result files grouped by dataset."""
    if not RESULTS_DIR.exists():
        return {}
    
    files_by_dataset = {}
    
    for f in RESULTS_DIR.glob("*.json"):
        if f.name.startswith('.') or 'test_history' in f.name:
            continue
        
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                metadata = data.get('metadata', {})
                file_dataset = metadata.get('dataset', {}).get('key', 'unknown')
                
                if file_dataset not in files_by_dataset:
                    files_by_dataset[file_dataset] = []
                
                files_by_dataset[file_dataset].append({
                    'path': f,
                    'name': f.name,
                    'model': metadata.get('model', 'unknown'),
                    'dataset': file_dataset,
                    'total': metadata.get('statistics', {}).get('total_questions', 0),
                    'run_id': metadata.get('run_id', '')
                })
        except:
            continue
    
    # Sort files within each dataset
    for dataset in files_by_dataset:
        files_by_dataset[dataset] = sorted(files_by_dataset[dataset], key=lambda x: x['name'], reverse=True)
    
    return files_by_dataset


def get_result_files_by_dataset(dataset_key):
    """Get list of result files for a specific dataset."""
    if not RESULTS_DIR.exists():
        return []
    
    files = []
    for f in RESULTS_DIR.glob("*.json"):
        if f.name.startswith('.') or 'test_history' in f.name:
            continue
        
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                metadata = data.get('metadata', {})
                file_dataset = metadata.get('dataset', {}).get('key', '')
                
                if file_dataset == dataset_key:
                    model = metadata.get('model', 'unknown')
                    total = metadata.get('statistics', {}).get('total_questions', 0)
                    run_id = metadata.get('run_id', '')
                    files.append({
                        'path': f,
                        'name': f.name,
                        'model': model,
                        'dataset': dataset_key,
                        'total': total,
                        'run_id': run_id
                    })
        except:
            continue
    
    return sorted(files, key=lambda x: x['name'], reverse=True)


def is_already_evaluated(result_file_name, llm_model, dataset_key):
    """Check if a result file has already been evaluated."""
    clean_llm = llm_model.replace(':', '_').replace('/', '_')
    eval_dir = EVALUATIONS_DIR / clean_llm / dataset_key
    
    if not eval_dir.exists():
        return False
    
    # Check if any evaluation file references this source file
    for eval_file in eval_dir.glob("*.json"):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('source_file') == result_file_name:
                    return True
        except:
            continue
    
    return False


def select_gemini_model():
    """Let user select which Gemini model to use for evaluation."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          🤖 SELECT GEMINI MODEL FOR EVALUATION             ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    for key, info in GEMINI_MODELS.items():
        recommended = " ⭐" if "recommended" in info['description'].lower() else ""
        print(f"  {Colors.BOLD}[{key}]{Colors.RESET} {Colors.GREEN}{info['name']}{Colors.RESET}{Colors.YELLOW}{recommended}{Colors.RESET}")
        print(f"      {Colors.DIM}{info['description']}{Colors.RESET}\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select model (1-{len(GEMINI_MODELS)}): {Colors.RESET}").strip()
            if choice in GEMINI_MODELS:
                selected = GEMINI_MODELS[choice]
                print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: {Colors.CYAN}{selected['name']}{Colors.RESET}")
                return selected['name']
            print(f"{Colors.RED}Invalid choice. Please enter 1-{len(GEMINI_MODELS)}.{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return None


def select_dataset():
    """Let user select which dataset to evaluate."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          📂 SELECT DATASET TO EVALUATE                     ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    for key, info in DATASETS.items():
        result_count = len(get_result_files_by_dataset(info['key']))
        status = f"{Colors.GREEN}[{result_count} result files]{Colors.RESET}" if result_count > 0 else f"{Colors.RED}[no results]{Colors.RESET}"
        
        print(f"  {Colors.BOLD}[{key}]{Colors.RESET} {Colors.YELLOW}{info['key'].upper()}{Colors.RESET} - {info['description']}")
        print(f"      {status}\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select dataset (1-{len(DATASETS)}): {Colors.RESET}").strip()
            if choice in DATASETS:
                selected = DATASETS[choice]
                print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: {Colors.CYAN}{selected['key'].upper()}{Colors.RESET}")
                return selected['key']
            print(f"{Colors.RED}Invalid choice. Please enter 1-{len(DATASETS)}.{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return None


def select_evaluation_mode():
    """Let user select single or batch evaluation mode."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          📋 SELECT EVALUATION MODE                         ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    print(f"  {Colors.BOLD}[1]{Colors.RESET} {Colors.GREEN}Single File{Colors.RESET}")
    print(f"      {Colors.DIM}Evaluate one result file{Colors.RESET}\n")
    
    print(f"  {Colors.BOLD}[2]{Colors.RESET} {Colors.YELLOW}Batch - All in Dataset{Colors.RESET}")
    print(f"      {Colors.DIM}Evaluate all result files for selected dataset{Colors.RESET}\n")
    
    print(f"  {Colors.BOLD}[3]{Colors.RESET} {Colors.MAGENTA}Batch - All Unevaluated{Colors.RESET}")
    print(f"      {Colors.DIM}Evaluate only files that haven't been evaluated yet{Colors.RESET}\n")
    
    print(f"  {Colors.BOLD}[4]{Colors.RESET} {Colors.CYAN}Batch - All Files (All Datasets){Colors.RESET}")
    print(f"      {Colors.DIM}Evaluate all result files across all datasets{Colors.RESET}\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select mode (1-4): {Colors.RESET}").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print(f"{Colors.RED}Invalid choice. Please enter 1-4.{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return None


def select_result_file(dataset_key):
    """Let user select a result file from the dataset."""
    files = get_result_files_by_dataset(dataset_key)
    
    if not files:
        print(f"\n{Colors.RED}No result files found for dataset '{dataset_key}'{Colors.RESET}")
        print(f"{Colors.DIM}Run a benchmark first using the Benchmark option.{Colors.RESET}")
        return None
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          📄 SELECT RESULT FILE                             ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    for idx, f in enumerate(files, 1):
        # Check if already evaluated
        already_eval = is_already_evaluated(f['name'], f['model'], dataset_key)
        eval_status = f"{Colors.GREEN}[evaluated]{Colors.RESET}" if already_eval else f"{Colors.YELLOW}[pending]{Colors.RESET}"
        
        print(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.GREEN}{f['name']}{Colors.RESET} {eval_status}")
        print(f"      {Colors.DIM}Model: {f['model']} | {f['total']} questions{Colors.RESET}\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Select file (1-{len(files)}): {Colors.RESET}").strip()
            idx = int(choice)
            if 1 <= idx <= len(files):
                selected = files[idx - 1]
                print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: {Colors.CYAN}{selected['name']}{Colors.RESET}")
                return selected
            print(f"{Colors.RED}Invalid choice. Please enter 1-{len(files)}.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return None


def select_result_files_multi(dataset_key):
    """Let user select multiple result files from the dataset."""
    files = get_result_files_by_dataset(dataset_key)
    
    if not files:
        print(f"\n{Colors.RED}No result files found for dataset '{dataset_key}'{Colors.RESET}")
        return []
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║          📄 SELECT RESULT FILES (comma-separated)          ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    for idx, f in enumerate(files, 1):
        already_eval = is_already_evaluated(f['name'], f['model'], dataset_key)
        eval_status = f"{Colors.GREEN}[evaluated]{Colors.RESET}" if already_eval else f"{Colors.YELLOW}[pending]{Colors.RESET}"
        
        print(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.GREEN}{f['name']}{Colors.RESET} {eval_status}")
        print(f"      {Colors.DIM}Model: {f['model']} | {f['total']} questions{Colors.RESET}\n")
    
    print(f"  {Colors.BOLD}[A]{Colors.RESET} Select ALL files")
    print(f"  {Colors.BOLD}[U]{Colors.RESET} Select only UNEVALUATED files\n")
    
    while True:
        try:
            choice = input(f"{Colors.CYAN}Enter selection (e.g., 1,2,3 or A or U): {Colors.RESET}").strip().upper()
            
            if choice == 'A':
                print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: ALL {len(files)} files")
                return files
            elif choice == 'U':
                unevaluated = [f for f in files if not is_already_evaluated(f['name'], f['model'], dataset_key)]
                print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: {len(unevaluated)} unevaluated files")
                return unevaluated
            else:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(files):
                        selected.append(files[idx - 1])
                if selected:
                    print(f"\n  {Colors.GREEN}✓{Colors.RESET} Selected: {len(selected)} files")
                    return selected
                print(f"{Colors.RED}Invalid selection.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter valid numbers separated by commas, or A/U.{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return []


def create_evaluation_prompt(result_data):
    """Create the system prompt and user prompt for batch evaluation."""
    
    metadata = result_data.get('metadata', {})
    results = result_data.get('results', [])
    model_tested = metadata.get('model', 'unknown')
    dataset_key = metadata.get('dataset', {}).get('key', 'unknown')
    
    evaluation_items = []
    for r in results:
        evaluation_items.append({
            "question_id": r.get('question_id', ''),
            "domain": r.get('domain', ''),
            "category": r.get('category', ''),
            "prompt": r.get('prompt', ''),
            "response": r.get('response', ''),
            "response_time": r.get('metrics', {}).get('response_time_seconds', 0)
        })
    
    system_prompt = """You are a strict LLM response evaluator. Your job is to score each response on a scale of 0-10.

## SCORING GUIDELINES (BE STRICT - most responses should score 4-7):

| Score | Rating | Criteria |
|-------|--------|----------|
| 0-2 | FAIL | Completely wrong, irrelevant, harmful, nonsensical, or refuses to answer |
| 3-4 | POOR | Major errors, significant omissions, misleading, or very incomplete |
| 5-6 | MEDIOCRE | Acceptable but has minor errors, could be improved, partially correct |
| 7-8 | GOOD | Mostly correct, helpful, clear, minor issues only |
| 9 | EXCELLENT | Near-perfect, comprehensive, accurate, well-structured |
| 10 | PERFECT | Flawless (EXTREMELY RARE - reserve for truly perfect answers) |

## IMPORTANT RULES:
1. Be STRICT - a score of 7+ means genuinely good quality
2. Score 10 should be given VERY RARELY - only for flawless responses
3. Consider the domain and category when evaluating
4. Brief but correct answers can still score high
5. Wrong answers should score 0-4 regardless of how well-written they are

## OUTPUT FORMAT:
You MUST respond with a valid JSON object containing an array of evaluations.
Each evaluation must have: question_id, score (integer 0-10), remark (brief explanation)

Example:
{
  "evaluations": [
    {"question_id": "Q1", "score": 7, "remark": "Correct answer with good explanation"},
    {"question_id": "Q2", "score": 3, "remark": "Wrong calculation, answer is incorrect"}
  ]
}"""

    user_prompt = f"""Evaluate the following {len(evaluation_items)} responses from the LLM model "{model_tested}" on the "{dataset_key}" dataset.

## RESPONSES TO EVALUATE:

```json
{json.dumps(evaluation_items, indent=2)}
```

Now evaluate each response and provide scores (0-10) with brief remarks. Respond ONLY with a valid JSON object."""

    return system_prompt, user_prompt, evaluation_items


def batch_evaluate(model, result_data, max_retries=3):
    """Evaluate all responses in a single API call."""
    
    system_prompt, user_prompt, evaluation_items = create_evaluation_prompt(result_data)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [{"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}]
            )
            
            text = response.text.strip()
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            eval_result = json.loads(text)
            evaluations = eval_result.get('evaluations', [])
            
            eval_map = {e['question_id']: e for e in evaluations}
            
            final_evaluations = []
            for item in evaluation_items:
                q_id = item['question_id']
                eval_data = eval_map.get(q_id, {})
                
                final_evaluations.append({
                    'question_id': q_id,
                    'prompt': item['prompt'],
                    'response': item['response'],
                    'response_time': item['response_time'],
                    'domain': item['domain'],
                    'category': item['category'],
                    'score': eval_data.get('score'),
                    'remark': eval_data.get('remark', 'No evaluation received')
                })
            
            return final_evaluations, None
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"
            if attempt < max_retries - 1:
                print(f"{Colors.YELLOW}  Retry {attempt + 2}/{max_retries} - JSON parse failed...{Colors.RESET}")
                time.sleep(2)
                continue
            return None, error_msg
            
        except Exception as e:
            error_msg = str(e)
            if 'quota' in error_msg.lower() or 'rate' in error_msg.lower() or '429' in error_msg:
                wait_time = (attempt + 1) * 15
                print(f"{Colors.YELLOW}  Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...{Colors.RESET}")
                time.sleep(wait_time)
                continue
            return None, error_msg
    
    return None, "Max retries exceeded"


def generate_output_path(llm_model, dataset_key, gemini_model):
    """Generate output path for evaluation results with proper naming."""
    clean_llm = llm_model.replace(':', '_').replace('/', '_')
    clean_gemini = gemini_model.replace('-', '_').replace('.', '_')
    
    output_dir = EVALUATIONS_DIR / clean_llm / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_llm}_{dataset_key}_{clean_gemini}_{timestamp}_eval.json"
    
    return output_dir / filename


def evaluate_single_file(result_info, gemini_model_name, model, delay_between=5):
    """Evaluate a single result file and return success status."""
    
    try:
        with open(result_info['path'], 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        metadata = result_data.get('metadata', {})
        results = result_data.get('results', [])
        llm_model = metadata.get('model', 'unknown')
        dataset_key = result_info['dataset']
        
        print(f"\n{Colors.CYAN}Processing: {result_info['name']}{Colors.RESET}")
        print(f"  {Colors.DIM}Model: {llm_model} | {len(results)} questions{Colors.RESET}")
        
        # Run evaluation
        start_time = time.time()
        evaluations, error = batch_evaluate(model, result_data)
        eval_time = time.time() - start_time
        
        if error:
            print(f"  {Colors.RED}✗ Failed: {error}{Colors.RESET}")
            return False, error
        
        # Calculate statistics
        scores = [e['score'] for e in evaluations if e['score'] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        evaluated_count = len(scores)
        
        # Save results
        output_path = generate_output_path(llm_model, dataset_key, gemini_model_name)
        
        output_data = {
            'evaluated_at': datetime.datetime.now().isoformat(),
            'source_file': result_info['name'],
            'llm_model': llm_model,
            'dataset': dataset_key,
            'evaluation_model': gemini_model_name,
            'evaluation_time_seconds': round(eval_time, 2),
            'total_evaluated': evaluated_count,
            'total_questions': len(evaluations),
            'average_score': round(avg_score, 2),
            'score_distribution': {
                'perfect_10': len([s for s in scores if s == 10]),
                'excellent_9': len([s for s in scores if s == 9]),
                'good_7_8': len([s for s in scores if 7 <= s <= 8]),
                'mediocre_5_6': len([s for s in scores if 5 <= s <= 6]),
                'poor_3_4': len([s for s in scores if 3 <= s <= 4]),
                'fail_0_2': len([s for s in scores if 0 <= s <= 2]),
            },
            'evaluations': evaluations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        score_color = Colors.GREEN if avg_score >= 7 else Colors.YELLOW if avg_score >= 4 else Colors.RED
        print(f"  {Colors.GREEN}✓{Colors.RESET} Completed in {eval_time:.1f}s | Avg Score: {score_color}{avg_score:.1f}/10{Colors.RESET}")
        print(f"  {Colors.DIM}Saved: {output_path.name}{Colors.RESET}")
        
        # Delay between evaluations to avoid rate limiting
        if delay_between > 0:
            time.sleep(delay_between)
        
        return True, None
        
    except Exception as e:
        print(f"  {Colors.RED}✗ Error: {str(e)}{Colors.RESET}")
        return False, str(e)


def run_batch_evaluation(files, gemini_model_name, delay_between=5):
    """Run batch evaluation on multiple files."""
    
    if not files:
        print(f"\n{Colors.YELLOW}No files to evaluate.{Colors.RESET}")
        return
    
    api_key = get_api_key()
    if not api_key:
        print(f"{Colors.RED}✗ GEMINI_API_KEY not found!{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║             🚀 BATCH EVALUATION STARTING                   ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"\n  Files to process: {Colors.CYAN}{len(files)}{Colors.RESET}")
    print(f"  Gemini model: {Colors.GREEN}{gemini_model_name}{Colors.RESET}")
    print(f"  Delay between files: {Colors.YELLOW}{delay_between}s{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    
    # Initialize Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(gemini_model_name)
    
    success_count = 0
    fail_count = 0
    total_start = time.time()
    
    for idx, file_info in enumerate(files, 1):
        print(f"\n{Colors.BOLD}[{idx}/{len(files)}]{Colors.RESET}", end="")
        success, error = evaluate_single_file(file_info, gemini_model_name, model, delay_between)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"""
{Colors.BOLD}{Colors.GREEN}
╔════════════════════════════════════════════════════════════╗
║                 ✓ BATCH EVALUATION COMPLETE                ║
╚════════════════════════════════════════════════════════════╝
{Colors.RESET}""")
    print(f"  Total Files:    {len(files)}")
    print(f"  {Colors.GREEN}Successful:     {success_count}{Colors.RESET}")
    print(f"  {Colors.RED}Failed:         {fail_count}{Colors.RESET}")
    print(f"  Total Time:     {total_time:.1f}s")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


def run_evaluation():
    """Main evaluation function with batch processing support."""
    clear_screen()
    
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   {Colors.YELLOW}███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗███████╗{Colors.CYAN}   ║
║   {Colors.YELLOW}██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██╔════╝{Colors.CYAN}   ║
║   {Colors.YELLOW}█████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   █████╗  {Colors.CYAN}   ║
║   {Colors.YELLOW}██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██╔══╝  {Colors.CYAN}   ║
║   {Colors.YELLOW}███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ███████╗{Colors.CYAN}   ║
║   {Colors.YELLOW}╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝{Colors.CYAN}   ║
║                                                                   ║
║        Batch LLM Evaluator - Strict 0-10 Scoring with Gemini      ║
║              Now with Batch Processing Support! 🚀                ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        print(f"{Colors.RED}✗ GEMINI_API_KEY not found!{Colors.RESET}")
        print(f"{Colors.DIM}Set it in .env file or as environment variable.{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")
        return
    
    print(f"{Colors.GREEN}✓{Colors.RESET} API key loaded")
    
    # Select evaluation mode
    mode = select_evaluation_mode()
    if not mode:
        return
    
    # Select Gemini model
    gemini_model_name = select_gemini_model()
    if not gemini_model_name:
        return
    
    if mode == '1':
        # Single file mode
        dataset_key = select_dataset()
        if not dataset_key:
            return
        
        result_info = select_result_file(dataset_key)
        if not result_info:
            input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")
            return
        
        # Run single evaluation
        run_batch_evaluation([result_info], gemini_model_name, delay_between=0)
        
    elif mode == '2':
        # Batch - All in dataset
        dataset_key = select_dataset()
        if not dataset_key:
            return
        
        files = get_result_files_by_dataset(dataset_key)
        if not files:
            print(f"\n{Colors.RED}No result files found for dataset '{dataset_key}'{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")
            return
        
        print(f"\n{Colors.YELLOW}Found {len(files)} files for {dataset_key.upper()}{Colors.RESET}")
        confirm = input(f"{Colors.CYAN}Proceed with batch evaluation? (y/n): {Colors.RESET}").strip().lower()
        if confirm == 'y':
            run_batch_evaluation(files, gemini_model_name)
        
    elif mode == '3':
        # Batch - Only unevaluated
        dataset_key = select_dataset()
        if not dataset_key:
            return
        
        files = get_result_files_by_dataset(dataset_key)
        unevaluated = [f for f in files if not is_already_evaluated(f['name'], f['model'], dataset_key)]
        
        if not unevaluated:
            print(f"\n{Colors.GREEN}All files in {dataset_key.upper()} have already been evaluated!{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")
            return
        
        print(f"\n{Colors.YELLOW}Found {len(unevaluated)} unevaluated files (out of {len(files)} total){Colors.RESET}")
        confirm = input(f"{Colors.CYAN}Proceed with batch evaluation? (y/n): {Colors.RESET}").strip().lower()
        if confirm == 'y':
            run_batch_evaluation(unevaluated, gemini_model_name)
        
    elif mode == '4':
        # Batch - All files across all datasets
        all_files = get_all_result_files()
        
        total_files = sum(len(files) for files in all_files.values())
        if total_files == 0:
            print(f"\n{Colors.RED}No result files found in any dataset.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")
            return
        
        print(f"\n{Colors.BOLD}Files by dataset:{Colors.RESET}")
        for dataset, files in all_files.items():
            unevaluated = [f for f in files if not is_already_evaluated(f['name'], f['model'], dataset)]
            print(f"  {Colors.YELLOW}{dataset.upper()}{Colors.RESET}: {len(files)} total, {len(unevaluated)} unevaluated")
        
        print(f"\n{Colors.BOLD}Options:{Colors.RESET}")
        print(f"  {Colors.BOLD}[A]{Colors.RESET} Evaluate ALL files ({total_files} total)")
        print(f"  {Colors.BOLD}[U]{Colors.RESET} Evaluate only UNEVALUATED files")
        print(f"  {Colors.BOLD}[C]{Colors.RESET} Cancel")
        
        choice = input(f"\n{Colors.CYAN}Select option: {Colors.RESET}").strip().upper()
        
        if choice == 'A':
            # Flatten all files
            flat_files = []
            for dataset, files in all_files.items():
                flat_files.extend(files)
            run_batch_evaluation(flat_files, gemini_model_name)
        elif choice == 'U':
            flat_files = []
            for dataset, files in all_files.items():
                for f in files:
                    if not is_already_evaluated(f['name'], f['model'], dataset):
                        flat_files.append(f)
            if flat_files:
                run_batch_evaluation(flat_files, gemini_model_name)
            else:
                print(f"\n{Colors.GREEN}All files have already been evaluated!{Colors.RESET}")
    
    input(f"\n{Colors.DIM}Press Enter to return...{Colors.RESET}")


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Evaluation interrupted.{Colors.RESET}")
        sys.exit(0)
