import json
from pathlib import Path
from typing import Dict
from knowornot.common.models import LabeledDataSample

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

def find_disagreements(evals_file: Path, samples_file: Path):
    """
    Find and print cases where GPT-4.1 disagrees with the combined human evaluation,
    focusing only on disagreements involving tier_3. Shows one disagreement at a time,
    waiting for user to press Enter before proceeding to the next.
    
    Args:
        evals_file: Path to the evaluation results JSON file
        samples_file: Path to the original labeled samples JSON file
    """
    # Load evaluation results
    evals_data = load_json_file(evals_file)
    if not evals_data:
        print("Failed to load evaluation data.")
        return
    
    # Extract model and human evaluations
    model_evals = evals_data.get('evaluations', {}).get('model', {})
    human_evals = evals_data.get('evaluations', {}).get('human', {}).get('combined', {})
    
    if not model_evals or not human_evals:
        print("Could not find model or human evaluations in the data.")
        return
    
    # Load the original samples
    try:
        labeled_samples = LabeledDataSample.load_list_from_json(samples_file)
        print(f"Successfully loaded {len(labeled_samples)} samples.")
    except Exception as e:
        print(f"Error loading labeled samples: {e}")
        return
    
    # Create a mapping of sample_id to sample object for quick lookup
    samples_dict = {sample.sample_id: sample for sample in labeled_samples}
    
    # Find disagreements that involve tier_3
    disagreements = []
    for sample_id, human_label in human_evals.items():
        model_label = model_evals.get(sample_id)
        
        if model_label and model_label != human_label:
            # Only include if either model or human classified as tier_3
            if model_label == 'tier_3' or human_label == 'tier_3':
                disagreements.append({
                    'sample_id': sample_id,
                    'model_label': model_label,
                    'human_label': human_label,
                    'sample': samples_dict.get(sample_id)
                })
    
    # Print initial count
    print(f"\nFound {len(disagreements)} tier_3-related disagreements between GPT-4.1 and combined human evaluation.")
    print("Press Enter to view each disagreement one by one...")
    input()
    
    # Display disagreements one by one, waiting for user to press Enter
    for i, item in enumerate(disagreements, 1):
        sample = item['sample']
        print("\n" + "=" * 80)
        print(f"Disagreement #{i} of {len(disagreements)} - Sample ID: {item['sample_id']}")
        print(f"Model classification: {item['model_label']} | Human classification: {item['human_label']}")
        print("-" * 80)
        
        if sample:
            print(f"Question: {sample.question}")
            print(f"\nExpected Answer: {sample.expected_answer}")
            print(f"\nLLM Response: {sample.llm_response.llm_response.response}")
            
            # Check for context questions
            if sample.context_questions:
                print("\nContext Questions:")
                for j, ctx_q in enumerate(sample.context_questions, 1):
                    print(f"  {j}. Question: {ctx_q.question}")
                    print(f"     Answer: {ctx_q.answer}")
                    
            # Check if this is the cited QA
            cited_qa = sample.llm_response.cited_QA
            if cited_qa:
                print("\nCited QA:")
                print(f"  Question: {cited_qa.question}")
                print(f"  Answer: {cited_qa.answer}")
        else:
            print(f"Warning: Could not find original sample data for ID: {item['sample_id']}")
        
        print("=" * 80)
        
        # If not the last item, wait for user to press Enter
        if i < len(disagreements):
            print("\nPress Enter to view the next disagreement...")
            input()
        else:
            print("\nThat was the last disagreement. Press Enter to see the summary...")
            input()
    
    # Print summary of tier_3 disagreements
    if disagreements:
        model_tier3_human_not = sum(1 for d in disagreements if d['model_label'] == 'tier_3' and d['human_label'] != 'tier_3')
        human_tier3_model_not = sum(1 for d in disagreements if d['human_label'] == 'tier_3' and d['model_label'] != 'tier_3')
        
        print("\nSummary of tier_3 Disagreements:")
        print("-" * 40)
        print(f"Model classified as tier_3 but humans did not: {model_tier3_human_not}")
        print(f"Humans classified as tier_3 but model did not: {human_tier3_model_not}")

if __name__ == "__main__":
    evals_file = Path("run_evaluations/factuality_evals_gpt_41.json")
    samples_file = Path("run_evaluations/label_samples_factuality_final.json")
    
    find_disagreements(evals_file, samples_file)