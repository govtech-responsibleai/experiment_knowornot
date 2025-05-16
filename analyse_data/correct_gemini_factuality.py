from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

# Import your models module
from knowornot.common.models import EvaluatedExperimentDocument

# Directory containing the JSON files
experiment_dir = Path("experiments/gemini_search_run_outputs")

# Dictionary to store our aggregated results
factuality_to_gemini_search = defaultdict(Counter)
gemini_search_to_factuality = defaultdict(Counter)

# Statistics counters
total_documents = 0
documents_with_required_evals = 0
total_responses = 0
responses_with_abstention_no = 0

print(f"Scanning directory: {experiment_dir}")
json_files = list(experiment_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

# Process all JSON files in the directory
for json_file in experiment_dir.glob("*.json"):
    try:
        print(f"Processing file: {json_file.name}")
        
        # Load the document
        doc = EvaluatedExperimentDocument.load_from_json(json_file)
        total_documents += 1
        
        # Check if the document has all required evaluations
        has_abstention = False
        has_factuality = False
        has_gemini_search = False
        
        eval_names = set()
        for eval_meta in doc.evaluation_metadata:
            eval_name = eval_meta.evaluation_name
            eval_names.add(eval_name)
            
            if eval_name == "abstention":
                has_abstention = True
            elif eval_name == "factuality":
                has_factuality = True
            elif eval_name == "gemini_search":
                has_gemini_search = True
        
        print(f"  Evaluations found: {eval_names}")
        
        if has_abstention and has_factuality and has_gemini_search:
            documents_with_required_evals += 1
            print("  Has all required evaluations")
            
            # Process each response in the document
            for response in doc.responses:
                total_responses += 1
                
                # Extract evaluation outcomes
                abstention_outcome = None
                factuality_outcome = None
                gemini_search_outcome = None
                
                for eval_output in response.evaluations:
                    if eval_output.evaluation_name == "abstention":
                        abstention_outcome = eval_output.evaluation_outcome
                    elif eval_output.evaluation_name == "factuality":
                        factuality_outcome = eval_output.evaluation_outcome
                    elif eval_output.evaluation_name == "gemini_search":
                        gemini_search_outcome = eval_output.evaluation_outcome
                
                # Only consider responses where abstention is "no"
                if abstention_outcome == "no":
                    responses_with_abstention_no += 1
                    
                    # Check if factuality is not "NA" when abstention is "no"
                    if factuality_outcome != "NA":
                        # Update our distribution counters
                        factuality_to_gemini_search[factuality_outcome][gemini_search_outcome] += 1
                        gemini_search_to_factuality[gemini_search_outcome][factuality_outcome] += 1
                    else:
                        print("  Warning: Found a response with abstention=no but factuality=NA")
        else:
            print("  Missing evaluations - needs abstention, factuality, and gemini_search")
        
    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total documents processed: {total_documents}")
print(f"Documents with all required evaluations: {documents_with_required_evals}")
print(f"Total responses: {total_responses}")
print(f"Responses with abstention=no: {responses_with_abstention_no}")

# Convert to DataFrame for better visualization
print("\n=== FACTUALITY TO GEMINI_SEARCH DISTRIBUTION ===")
factuality_df = pd.DataFrame.from_dict(
    {k: dict(v) for k, v in factuality_to_gemini_search.items()}, 
    orient='index'
).fillna(0)
print(factuality_df)
print("\nPercentages:")
factuality_percentage = factuality_df.div(factuality_df.sum(axis=1), axis=0) * 100
print(factuality_percentage.round(2))

print("\n=== GEMINI_SEARCH TO FACTUALITY DISTRIBUTION ===")
sg_policy_df = pd.DataFrame.from_dict(
    {k: dict(v) for k, v in gemini_search_to_factuality.items()}, 
    orient='index'
).fillna(0)
print(sg_policy_df)
print("\nPercentages:")
sg_policy_percentage = sg_policy_df.div(sg_policy_df.sum(axis=1), axis=0) * 100
print(sg_policy_percentage.round(2))

# List out the unique values found for each evaluation type
print("\n=== UNIQUE VALUES FOUND ===")
print(f"Factuality values: {list(factuality_to_gemini_search.keys())}")
print(f"Gemini Search values: {list(gemini_search_to_factuality.keys())}")