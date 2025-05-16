import json
from pathlib import Path
from typing import Dict, List

def load_json_file(file_path: Path) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return {}

def analyze_model_performance(eval_file: Path, human_annotator: str = "combined"):
    """
    Analyzes model performance treating tier_1 and tier_2 as the same category.
    
    Args:
        eval_file: Path to the evaluation results JSON file
        human_annotator: The human annotator to compare against
    
    Returns:
        Dict with confusion matrix and metrics
    """
    # Load evaluation results
    evals_data = load_json_file(eval_file)
    if not evals_data:
        print(f"Failed to load evaluation data from {eval_file}")
        return None
    
    # Get model name from metadata
    model_name = evals_data.get('metadata', {}).get('model', f"Unknown Model ({eval_file.name})")
    
    # Extract model and human evaluations
    model_evals = evals_data.get('evaluations', {}).get('model', {})
    human_evals = evals_data.get('evaluations', {}).get('human', {}).get(human_annotator, {})
    
    if not model_evals or not human_evals:
        print(f"Could not find model or human evaluations in {eval_file}")
        return None
    
    # Initialize confusion matrix (treating tier_1 and tier_2 as "acceptable")
    confusion_matrix = {
        "TP": 0,  # True Positive: Both model and human say tier_3
        "FP": 0,  # False Positive: Model says tier_3, human says acceptable
        "FN": 0,  # False Negative: Model says acceptable, human says tier_3
        "TN": 0,  # True Negative: Both model and human say acceptable
    }
    
    # Categorize each sample
    total_samples = 0
    
    for sample_id, human_label in human_evals.items():
        if sample_id in model_evals:
            total_samples += 1
            model_label = model_evals[sample_id]
            
            # Convert to binary classification
            human_binary = "tier_3" if human_label == "tier_3" else "acceptable"
            model_binary = "tier_3" if model_label == "tier_3" else "acceptable"
            
            # Update confusion matrix
            if human_binary == "tier_3" and model_binary == "tier_3":
                confusion_matrix["TP"] += 1
            elif human_binary == "acceptable" and model_binary == "tier_3":
                confusion_matrix["FP"] += 1
            elif human_binary == "tier_3" and model_binary == "acceptable":
                confusion_matrix["FN"] += 1
            else:  # human_binary == "acceptable" and model_binary == "acceptable"
                confusion_matrix["TN"] += 1
    
    # Calculate metrics
    tp = confusion_matrix["TP"]
    fp = confusion_matrix["FP"]
    fn = confusion_matrix["FN"]
    tn = confusion_matrix["TN"]
    
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        "model_name": model_name,
        "confusion_matrix": confusion_matrix,
        "total_samples": total_samples,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,  # PPV: When model predicts tier_3, how often is it right
            "recall": recall,        # Sensitivity: How many actual tier_3 cases does model catch
            "f1_score": f1_score,
            "specificity": specificity,  # When it's actually acceptable, how often does model get it right
            "npv": npv,              # When model predicts acceptable, how often is it right
        }
    }

def print_model_analysis(analysis):
    """Prints the analysis results in a readable format."""
    if not analysis:
        return
    
    cm = analysis["confusion_matrix"]
    metrics = analysis["metrics"]
    
    print(f"\n===== {analysis['model_name']} =====")
    print(f"Total samples analyzed: {analysis['total_samples']}")
    
    print("\nConfusion Matrix (treating tier_1 and tier_2 as 'acceptable'):")
    print("                  | Human: tier_3 | Human: acceptable |")
    print(f"Model: tier_3     | TP: {cm['TP']:8d} | FP: {cm['FP']:15d} |")
    print(f"Model: acceptable | FN: {cm['FN']:8d} | TN: {cm['TN']:15d} |")
    
    # Calculate percentages for each cell
    total = analysis["total_samples"]
    tp_pct = cm['TP'] / total * 100 if total > 0 else 0
    fp_pct = cm['FP'] / total * 100 if total > 0 else 0
    fn_pct = cm['FN'] / total * 100 if total > 0 else 0
    tn_pct = cm['TN'] / total * 100 if total > 0 else 0
    
    print("\nAs Percentages of Total:")
    print("                  | Human: tier_3 | Human: acceptable |")
    print(f"Model: tier_3     | TP: {tp_pct:8.2f}% | FP: {fp_pct:15.2f}% |")
    print(f"Model: acceptable | FN: {fn_pct:8.2f}% | TN: {tn_pct:15.2f}% |")
    
    print("\nPerformance Metrics:")
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:   {metrics['precision'] * 100:.2f}% (When model says tier_3, how often is it right)")
    print(f"Recall:      {metrics['recall'] * 100:.2f}% (What % of actual tier_3 cases does model catch)")
    print(f"F1 Score:    {metrics['f1_score'] * 100:.2f}%")
    print(f"Specificity: {metrics['specificity'] * 100:.2f}% (When it's actually acceptable, how often does model get it right)")
    print(f"NPV:         {metrics['npv'] * 100:.2f}% (When model says acceptable, how often is it right)")
    
    # Add interpretations
    if metrics['precision'] > metrics['recall']:
        print("\nThis model is more precise than sensitive - it's careful about labeling things as tier_3,")
        print("but may miss some problematic responses.")
    elif metrics['recall'] > metrics['precision']:
        print("\nThis model is more sensitive than precise - it catches most problematic responses,")
        print("but may incorrectly flag some acceptable responses as tier_3.")
    
    # Calculate error types
    over_strict = cm['FP'] / (cm['FP'] + cm['TN']) * 100 if (cm['FP'] + cm['TN']) > 0 else 0
    over_lenient = cm['FN'] / (cm['TP'] + cm['FN']) * 100 if (cm['TP'] + cm['FN']) > 0 else 0
    
    print(f"\nOver-strictness: {over_strict:.2f}% of acceptable responses were flagged as tier_3")
    print(f"Over-leniency:  {over_lenient:.2f}% of tier_3 responses were classified as acceptable")

def compare_models(eval_files: List[Path], human_annotator: str = "combined"):
    """Compare metrics across multiple models."""
    results = []
    
    for file_path in eval_files:
        analysis = analyze_model_performance(file_path, human_annotator)
        if analysis:
            results.append(analysis)
            print_model_analysis(analysis)
    
    if len(results) > 1:
        print("\n===== Model Comparison Summary =====")
        print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'FP %':<10} | {'FN %':<10}")
        print("-" * 85)
        
        for r in results:
            metrics = r["metrics"]
            cm = r["confusion_matrix"]
            total = r["total_samples"]
            
            fp_pct = cm['FP'] / total * 100 if total > 0 else 0
            fn_pct = cm['FN'] / total * 100 if total > 0 else 0
            
            print(f"{r['model_name']:<25} | {metrics['accuracy']*100:8.2f}% | {metrics['precision']*100:8.2f}% | {metrics['recall']*100:8.2f}% | {metrics['f1_score']*100:8.2f}% | {fp_pct:8.2f}% | {fn_pct:8.2f}%")

if __name__ == "__main__":
    eval_files = [
        Path("run_evaluations/factuality_evals_gpt_41.json"),
        Path("run_evaluations/factuality_evals_gemini_2_5_flash.json"),
        Path("run_evaluations/factuality_evals_gemini_2_flash.json"),
        Path("run_evaluations/factuality_evals_gemini_2_5_pro.json"),
        Path("run_evaluations/factuality_evals_o4_mini.json")
    ]
    
    compare_models(eval_files)