from pathlib import Path
from typing import List
import pandas as pd
from knowornot.common.models import EvaluatedExperimentDocument, EvaluationOutput

def load_all_evaluated_experiment_documents(directory_path: str = "experiments/factuality_run_outputs") -> List[EvaluatedExperimentDocument]:
    """
    Load all EvaluatedExperimentDocuments from JSON files in the specified directory.
    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    json_files = list(path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {path}")
        return []
    
    documents = []
    for json_file in json_files:
        try:
            document = EvaluatedExperimentDocument.load_from_json(json_file)
            documents.append(document)
            print(f"Successfully loaded: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {str(e)}")
    
    print(f"Loaded {len(documents)} documents")
    return documents

def get_evaluation_outcome_by_name(evaluations: List[EvaluationOutput], eval_name: str) -> str:
    """
    Get the evaluation outcome for a specific evaluation name.
    
    Args:
        evaluations: List of evaluation outputs
        eval_name: Name of the evaluation to find
        
    Returns:
        The evaluation outcome or "Not found" if not present
    """
    for eval_output in evaluations:
        if eval_output.evaluation_name == eval_name:
            return eval_output.evaluation_outcome
    return "Not found"

def create_evaluation_csv(documents: List[EvaluatedExperimentDocument], output_file: str = "evaluation_results.csv"):
    """
    Create a CSV with factuality and abstention evaluation results.
    """
    # Define columns for the CSV
    columns = [
        "document_filename", 
        "experiment_type", 
        "knowledge_base_identifier", 
        "retrieval_type", 
        "llm_model", 
        "system_prompt_name",
        "question_id", 
        "response_id", 
        "factuality", 
        "abstention"
    ]
    
    # Prepare data for the CSV
    rows = []
    
    for doc in documents:
        # Get document-level information
        doc_filename = doc.path_to_store.name
        experiment_type = doc.experiment_metadata.experiment_type.value
        knowledge_base = doc.experiment_metadata.knowledge_base_identifier
        retrieval_type = doc.experiment_metadata.retrieval_type.value
        llm_model = doc.experiment_metadata.ai_model_used
        system_prompt_name = doc.experiment_metadata.system_prompt.identifier
        
        # Process each response
        for response in doc.responses:
            # Get response identifiers
            question_id = response.llm_response.identifier
            response_id = response.llm_response.llm_response.response[:10].strip()  # Using first 10 chars as ID
            
            # Get evaluation outcomes
            factuality = get_evaluation_outcome_by_name(response.evaluations, "factuality")
            abstention = get_evaluation_outcome_by_name(response.evaluations, "abstention")
            
            # Create row and add to rows list
            row = {
                "document_filename": doc_filename,
                "experiment_type": experiment_type,
                "knowledge_base_identifier": knowledge_base,
                "retrieval_type": retrieval_type,
                "llm_model": llm_model,
                "system_prompt_name": system_prompt_name,
                "question_id": question_id,
                "response_id": response_id,
                "factuality": factuality,
                "abstention": abstention
            }
            rows.append(row)
    
    # Create and save the CSV
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"CSV file created at {output_file} with {len(rows)} rows")

if __name__ == "__main__":
    # Load all documents
    documents = load_all_evaluated_experiment_documents()
    
    # Create the CSV file
    create_evaluation_csv(documents)
    
    print("Process completed successfully!")