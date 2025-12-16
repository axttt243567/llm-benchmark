#!/usr/bin/env python3
"""
LLM Benchmark - Advanced Unified CLI
=====================================
A full-featured command-line interface for the LLM Benchmark Pipeline.

Features:
  1. Benchmark - Test LLM models with question datasets
  2. Evaluate  - Batch score responses using Gemini AI (0-10)
  3. Visualize - Interactive dashboard to analyze results
  4. Cleanup   - Manage and delete old files
  5. Status    - View current files and statistics

Usage: python scripts/cli.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Directories
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
DATA_DIR = PROJECT_ROOT / "data"

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_terminal_width():
    """Get terminal width."""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80


def print_header(title, emoji="🧠"):
    """Print a stylized header."""
    width = min(get_terminal_width() - 4, 75)
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔{'═' * width}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}  {emoji} {title}{' ' * (width - len(title) - 4)}{Colors.BOLD}{Colors.CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * width}╝{Colors.RESET}\n")


def print_divider():
    """Print a divider line."""
    width = min(get_terminal_width() - 4, 75)
    print(f"{Colors.DIM}{'─' * width}{Colors.RESET}")


# ============================================================================
# FILE MANAGEMENT UTILITIES
# ============================================================================

def get_all_result_files():
    """Get all result JSON files."""
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
                files.append({
                    'path': f,
                    'name': f.name,
                    'model': metadata.get('model', 'unknown'),
                    'dataset': metadata.get('dataset', {}).get('key', 'unknown'),
                    'total': metadata.get('statistics', {}).get('total_questions', 0),
                    'date': metadata.get('execution', {}).get('start_time', '')[:10],
                    'size': f.stat().st_size
                })
        except:
            files.append({
                'path': f,
                'name': f.name,
                'model': 'unknown',
                'dataset': 'unknown',
                'total': 0,
                'date': '',
                'size': f.stat().st_size
            })
    
    return sorted(files, key=lambda x: x['name'], reverse=True)


def get_all_evaluation_folders():
    """Get all evaluation folders with their contents."""
    if not EVALUATIONS_DIR.exists():
        return []
    
    folders = []
    for model_dir in EVALUATIONS_DIR.iterdir():
        if model_dir.is_dir():
            for dataset_dir in model_dir.iterdir():
                if dataset_dir.is_dir():
                    eval_files = list(dataset_dir.glob("*.json"))
                    if eval_files:
                        total_size = sum(f.stat().st_size for f in eval_files)
                        folders.append({
                            'path': dataset_dir,
                            'model': model_dir.name,
                            'dataset': dataset_dir.name,
                            'file_count': len(eval_files),
                            'total_size': total_size
                        })
    
    return sorted(folders, key=lambda x: f"{x['model']}_{x['dataset']}")


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


# ============================================================================
# MAIN MENU
# ============================================================================

def display_main_menu():
    """Display the main menu."""
    clear_screen()
    
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   {Colors.YELLOW}██╗     ██╗     ███╗   ███╗    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗{Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ████╗ ████║    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║{Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ██╔████╔██║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║{Colors.CYAN}   ║
║   {Colors.YELLOW}██║     ██║     ██║╚██╔╝██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║{Colors.CYAN}   ║
║   {Colors.YELLOW}███████╗███████╗██║ ╚═╝ ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║{Colors.CYAN}   ║
║   {Colors.YELLOW}╚══════╝╚══════╝╚═╝     ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝{Colors.CYAN}   ║
║                                                                           ║
║                    Advanced LLM Benchmark Pipeline v3.0                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")
    
    # Get stats
    result_count = len(get_all_result_files())
    eval_folders = get_all_evaluation_folders()
    eval_count = sum(f['file_count'] for f in eval_folders)
    
    print(f"  {Colors.DIM}Results: {result_count} files | Evaluations: {eval_count} files{Colors.RESET}\n")
    print_divider()
    
    options = [
        ("1", "🧠 Benchmark", "Test LLM models with question datasets", Colors.CYAN),
        ("2", "📊 Evaluate", "Batch score responses using Gemini AI", Colors.MAGENTA),
        ("3", "📈 Visualize", "Interactive dashboard for analysis", Colors.GREEN),
        ("4", "🧹 Cleanup", "Manage and delete old files", Colors.YELLOW),
        ("5", "📋 Status", "View current files and statistics", Colors.BLUE),
        ("0", "🚪 Exit", "Exit the application", Colors.RED),
    ]
    
    print()
    for key, title, desc, color in options:
        print(f"  {Colors.BOLD}{color}[{key}]{Colors.RESET} {Colors.WHITE}{title}{Colors.RESET}")
        print(f"      {Colors.DIM}{desc}{Colors.RESET}\n")
    
    print_divider()


# ============================================================================
# CLEANUP MENU
# ============================================================================

def display_cleanup_menu():
    """Display the cleanup submenu."""
    clear_screen()
    print_header("CLEANUP & FILE MANAGEMENT", "🧹")
    
    # Show current stats
    result_files = get_all_result_files()
    eval_folders = get_all_evaluation_folders()
    
    total_result_size = sum(f['size'] for f in result_files)
    total_eval_size = sum(f['total_size'] for f in eval_folders)
    
    print(f"  {Colors.BOLD}Current Files:{Colors.RESET}")
    print(f"    Results:     {Colors.CYAN}{len(result_files)} files{Colors.RESET} ({format_size(total_result_size)})")
    print(f"    Evaluations: {Colors.CYAN}{len(eval_folders)} folders{Colors.RESET} ({format_size(total_eval_size)})")
    
    # Check for test history
    test_history = RESULTS_DIR / ".test_history.json"
    if test_history.exists():
        print(f"    Test History: {Colors.YELLOW}exists{Colors.RESET} ({format_size(test_history.stat().st_size)})")
    
    print()
    print_divider()
    
    options = [
        ("1", "📄 Delete Result Files", "Select specific result files to delete"),
        ("2", "📂 Delete Evaluation Folders", "Select specific evaluation folders to delete"),
        ("3", "🗑️  Clear All Results", "Delete ALL result files"),
        ("4", "🗑️  Clear All Evaluations", "Delete ALL evaluation files"),
        ("5", "📜 Clear Test History", "Delete .test_history.json"),
        ("6", "💥 Clear Everything", "Delete all results, evaluations, and history"),
        ("0", "⬅️  Back", "Return to main menu"),
    ]
    
    print()
    for key, title, desc in options:
        color = Colors.RED if key in ['3', '4', '6'] else Colors.YELLOW if key == '5' else Colors.CYAN
        print(f"  {Colors.BOLD}{color}[{key}]{Colors.RESET} {title}")
        print(f"      {Colors.DIM}{desc}{Colors.RESET}\n")
    
    print_divider()


def select_files_to_delete(files, file_type="files"):
    """Interactive file selection for deletion."""
    if not files:
        print(f"\n{Colors.YELLOW}No {file_type} found.{Colors.RESET}")
        input(f"{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        return []
    
    print(f"\n{Colors.BOLD}Available {file_type}:{Colors.RESET}\n")
    
    for idx, f in enumerate(files, 1):
        if 'file_count' in f:  # Evaluation folder
            print(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.CYAN}{f['model']}/{f['dataset']}{Colors.RESET}")
            print(f"      {Colors.DIM}{f['file_count']} files, {format_size(f['total_size'])}{Colors.RESET}")
        else:  # Result file
            print(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {Colors.CYAN}{f['name']}{Colors.RESET}")
            print(f"      {Colors.DIM}{f['model']} | {f['dataset']} | {f['total']} Q | {format_size(f['size'])}{Colors.RESET}")
    
    print(f"\n{Colors.DIM}Enter numbers separated by commas (e.g., 1,3,5) or 'all' for everything{Colors.RESET}")
    print(f"{Colors.DIM}Enter '0' or 'cancel' to abort{Colors.RESET}")
    
    selection = input(f"\n{Colors.CYAN}Select {file_type} to delete: {Colors.RESET}").strip().lower()
    
    if selection in ['0', 'cancel', '']:
        print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
        return []
    
    if selection == 'all':
        return files
    
    try:
        indices = [int(x.strip()) for x in selection.split(',')]
        selected = [files[i-1] for i in indices if 1 <= i <= len(files)]
        return selected
    except (ValueError, IndexError):
        print(f"{Colors.RED}Invalid selection.{Colors.RESET}")
        return []


def confirm_deletion(items, item_type="items"):
    """Confirm deletion with 'yes' requirement."""
    if not items:
        return False
    
    print(f"\n{Colors.BOLD}{Colors.RED}⚠️  WARNING: This will permanently delete {len(items)} {item_type}!{Colors.RESET}")
    print(f"{Colors.DIM}This action cannot be undone.{Colors.RESET}\n")
    
    for item in items[:5]:
        if 'file_count' in item:
            print(f"  • {item['model']}/{item['dataset']} ({item['file_count']} files)")
        else:
            print(f"  • {item['name']}")
    
    if len(items) > 5:
        print(f"  {Colors.DIM}... and {len(items) - 5} more{Colors.RESET}")
    
    confirm = input(f"\n{Colors.RED}Type 'yes' to confirm deletion: {Colors.RESET}").strip().lower()
    return confirm == 'yes'


def delete_result_files(files):
    """Delete selected result files."""
    for f in files:
        try:
            f['path'].unlink()
            print(f"  {Colors.GREEN}✓{Colors.RESET} Deleted {f['name']}")
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.RESET} Failed to delete {f['name']}: {e}")


def delete_evaluation_folders(folders):
    """Delete selected evaluation folders."""
    for folder in folders:
        try:
            shutil.rmtree(folder['path'])
            print(f"  {Colors.GREEN}✓{Colors.RESET} Deleted {folder['model']}/{folder['dataset']}")
            
            # Clean up empty parent directories
            parent = folder['path'].parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                print(f"  {Colors.DIM}  (removed empty folder {parent.name}){Colors.RESET}")
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.RESET} Failed to delete {folder['model']}/{folder['dataset']}: {e}")


def run_cleanup_menu():
    """Run the cleanup submenu."""
    while True:
        display_cleanup_menu()
        
        choice = input(f"  {Colors.CYAN}Enter choice: {Colors.RESET}").strip()
        
        if choice == '0':
            return
        
        elif choice == '1':  # Delete specific result files
            clear_screen()
            print_header("DELETE RESULT FILES", "📄")
            files = get_all_result_files()
            selected = select_files_to_delete(files, "result files")
            if selected and confirm_deletion(selected, "result files"):
                delete_result_files(selected)
                print(f"\n{Colors.GREEN}✓ Deleted {len(selected)} result files.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        
        elif choice == '2':  # Delete specific evaluation folders
            clear_screen()
            print_header("DELETE EVALUATION FOLDERS", "📂")
            folders = get_all_evaluation_folders()
            selected = select_files_to_delete(folders, "evaluation folders")
            if selected and confirm_deletion(selected, "evaluation folders"):
                delete_evaluation_folders(selected)
                print(f"\n{Colors.GREEN}✓ Deleted {len(selected)} evaluation folders.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        
        elif choice == '3':  # Clear all results
            clear_screen()
            print_header("CLEAR ALL RESULTS", "🗑️")
            files = get_all_result_files()
            if files and confirm_deletion(files, "result files"):
                delete_result_files(files)
                print(f"\n{Colors.GREEN}✓ Cleared all result files.{Colors.RESET}")
            elif not files:
                print(f"{Colors.YELLOW}No result files to delete.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        
        elif choice == '4':  # Clear all evaluations
            clear_screen()
            print_header("CLEAR ALL EVALUATIONS", "🗑️")
            folders = get_all_evaluation_folders()
            if folders and confirm_deletion(folders, "evaluation folders"):
                delete_evaluation_folders(folders)
                # Remove evaluations directory if empty
                if EVALUATIONS_DIR.exists() and not any(EVALUATIONS_DIR.iterdir()):
                    EVALUATIONS_DIR.rmdir()
                print(f"\n{Colors.GREEN}✓ Cleared all evaluations.{Colors.RESET}")
            elif not folders:
                print(f"{Colors.YELLOW}No evaluations to delete.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        
        elif choice == '5':  # Clear test history
            clear_screen()
            print_header("CLEAR TEST HISTORY", "📜")
            history_file = RESULTS_DIR / ".test_history.json"
            if history_file.exists():
                print(f"\n{Colors.YELLOW}This will reset all test counters.{Colors.RESET}")
                confirm = input(f"{Colors.RED}Type 'yes' to confirm: {Colors.RESET}").strip().lower()
                if confirm == 'yes':
                    history_file.unlink()
                    print(f"\n{Colors.GREEN}✓ Test history cleared.{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}No test history file found.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
        
        elif choice == '6':  # Clear everything
            clear_screen()
            print_header("CLEAR EVERYTHING", "💥")
            print(f"\n{Colors.BOLD}{Colors.RED}⚠️  DANGER ZONE ⚠️{Colors.RESET}")
            print(f"{Colors.RED}This will delete ALL results, evaluations, and test history!{Colors.RESET}\n")
            
            result_files = get_all_result_files()
            eval_folders = get_all_evaluation_folders()
            history_exists = (RESULTS_DIR / ".test_history.json").exists()
            
            print(f"  • {len(result_files)} result files")
            print(f"  • {len(eval_folders)} evaluation folders")
            print(f"  • Test history: {'Yes' if history_exists else 'No'}")
            
            confirm = input(f"\n{Colors.RED}Type 'DELETE EVERYTHING' to confirm: {Colors.RESET}").strip()
            if confirm == 'DELETE EVERYTHING':
                if result_files:
                    delete_result_files(result_files)
                if eval_folders:
                    delete_evaluation_folders(eval_folders)
                if history_exists:
                    (RESULTS_DIR / ".test_history.json").unlink()
                print(f"\n{Colors.GREEN}✓ Everything cleared!{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# STATUS VIEW
# ============================================================================

def display_status():
    """Display detailed status of all files."""
    clear_screen()
    print_header("STATUS & STATISTICS", "📋")
    
    # Results section
    result_files = get_all_result_files()
    print(f"{Colors.BOLD}📄 Result Files ({len(result_files)}):{Colors.RESET}")
    print_divider()
    
    if result_files:
        for f in result_files[:10]:
            print(f"  {Colors.CYAN}{f['name']}{Colors.RESET}")
            print(f"    {Colors.DIM}Model: {f['model']} | Dataset: {f['dataset']} | {f['total']} questions | {format_size(f['size'])}{Colors.RESET}")
        if len(result_files) > 10:
            print(f"  {Colors.DIM}... and {len(result_files) - 10} more files{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}No result files found{Colors.RESET}")
    
    print()
    
    # Evaluations section
    eval_folders = get_all_evaluation_folders()
    total_eval_files = sum(f['file_count'] for f in eval_folders)
    print(f"{Colors.BOLD}📊 Evaluation Folders ({len(eval_folders)} folders, {total_eval_files} files):{Colors.RESET}")
    print_divider()
    
    if eval_folders:
        for folder in eval_folders:
            print(f"  {Colors.MAGENTA}{folder['model']}/{folder['dataset']}{Colors.RESET}")
            print(f"    {Colors.DIM}{folder['file_count']} files, {format_size(folder['total_size'])}{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}No evaluation folders found{Colors.RESET}")
    
    print()
    
    # Dataset files - organized in category folders
    print(f"{Colors.BOLD}📁 Available Datasets:{Colors.RESET}")
    print_divider()
    
    if DATA_DIR.exists():
        for folder in sorted(DATA_DIR.iterdir()):
            if folder.is_dir():
                print(f"  {Colors.CYAN}📂 {folder.name}/{Colors.RESET}")
                for f in sorted(folder.glob("*.json")):
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            questions = json.load(file)
                            count = len(questions)
                        print(f"    {Colors.GREEN}{f.name}{Colors.RESET} - {count} questions")
                    except:
                        print(f"    {Colors.YELLOW}{f.name}{Colors.RESET} - error reading")
    
    print()
    input(f"{Colors.DIM}Press Enter to return to menu...{Colors.RESET}")


# ============================================================================
# MODULE RUNNERS
# ============================================================================

def run_benchmark():
    """Run the benchmark module."""
    try:
        from benchmark import run_benchmark as benchmark_main
        benchmark_main()
    except ImportError as e:
        print(f"{Colors.RED}Error importing benchmark module: {e}{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error running benchmark: {e}{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def run_evaluation():
    """Run the evaluation module."""
    try:
        from evaluate import run_evaluation as evaluate_main
        evaluate_main()
    except ImportError as e:
        print(f"{Colors.RED}Error importing evaluate module: {e}{Colors.RESET}")
        print(f"{Colors.DIM}Make sure google-generativeai is installed:{Colors.RESET}")
        print(f"{Colors.CYAN}  pip install google-generativeai python-dotenv{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error running evaluation: {e}{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


def run_visualization():
    """Run the visualization dashboard."""
    try:
        from visualize import main as visualize_main
        visualize_main()
    except ImportError as e:
        print(f"{Colors.RED}Error importing visualize module: {e}{Colors.RESET}")
        print(f"{Colors.DIM}Make sure required packages are installed:{Colors.RESET}")
        print(f"{Colors.CYAN}  pip install dash dash-bootstrap-components plotly pandas{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error running visualization: {e}{Colors.RESET}")
        input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the unified CLI."""
    while True:
        display_main_menu()
        
        try:
            choice = input(f"  {Colors.CYAN}Enter choice: {Colors.RESET}").strip()
            
            if choice == '1':
                run_benchmark()
            elif choice == '2':
                run_evaluation()
            elif choice == '3':
                run_visualization()
            elif choice == '4':
                run_cleanup_menu()
            elif choice == '5':
                display_status()
            elif choice == '0':
                clear_screen()
                print(f"\n{Colors.GREEN}Thank you for using LLM Benchmark Pipeline!{Colors.RESET}")
                print(f"{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                sys.exit(0)
            else:
                print(f"\n{Colors.RED}Invalid choice. Please enter 0-5.{Colors.RESET}")
                input(f"{Colors.DIM}Press Enter to continue...{Colors.RESET}")
                
        except KeyboardInterrupt:
            clear_screen()
            print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.RESET}")
            print(f"{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
