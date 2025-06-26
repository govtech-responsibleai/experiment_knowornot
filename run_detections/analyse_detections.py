import argparse
from pathlib import Path
import pandas as pd
from knowornot.Detector.models import DetectedExperimentDocument, DetectionStatus, DetectionLabel, DetectorType
from knowornot.common.models import EvaluatedExperimentDocument
import itertools

def get_all_detector_types(detected_dir):
    detector_types = set()
    for file in Path(detected_dir).glob("*.json"):
        try:
            doc = DetectedExperimentDocument.load_from_json(file)
            for response in doc.responses:
                for detection in response.detections:
                    detector_types.add(str(detection.detector_type))
        except Exception as e:
            print(f"Error loading detection file {file}: {e}")
    return sorted(detector_types)

def extract_detection_rows(detected_dir, detector_types):
    rows = []
    for file in Path(detected_dir).glob("*.json"):
        try:
            doc = DetectedExperimentDocument.load_from_json(file)
            meta = doc.experiment_metadata
            for response in doc.responses:
                llm = response.llm_response
                row = {
                    "llm_response_identifier": llm.identifier,
                    "question": getattr(llm.experiment_input.source_context_qa, "question", None),
                    "model_answer": getattr(llm.llm_response, "response", None),
                    "expected_answer": getattr(llm.experiment_input.source_context_qa, "expected_answer", None),
                    "knowledge_base_identifier": meta.get("knowledge_base_identifier"),
                    "experiment_type": meta.get("experiment_type"),
                    "system_prompt_identifier": meta.get("system_prompt", {}).get("identifier") if isinstance(meta.get("system_prompt"), dict) else None,
                    "retrieval_type": meta.get("retrieval_type"),
                    "ai_model_used": meta.get("ai_model_used"),
                }
                # Add detection columns for each detector type
                detection_map = {str(d.dector_type if hasattr(d, 'dector_type') else d.detector_type): d for d in response.detections}
                for det_type in detector_types:
                    det = detection_map.get(det_type)
                    # Save .name if value is an enum
                    label = getattr(det, "label", None) if det else None
                    if hasattr(label, "name"):
                        label = label.name
                    status = getattr(det, "status", None) if det else None
                    if hasattr(status, "name"):
                        status = status.name
                    dtype = getattr(det, "detector_type", None) if det else None
                    if hasattr(dtype, "name"):
                        dtype = dtype.name
                    row[f"{dtype}_label"] = label
                    row[f"{dtype}_score"] = getattr(det, "confidence", None) if det else None
                    row[f"{dtype}_explanation"] = getattr(det, "explanation", None) if det else None
                    row[f"{dtype}_status"] = status
                rows.append(row)
        except Exception as e:
            print(f"Error loading detection file {file}: {e}")
    return rows

def extract_evaluation_rows(evaluated_doc):
    eval_names = [em.evaluation_name for em in getattr(evaluated_doc, "evaluation_metadata", [])]
    rows = {}
    for response in evaluated_doc.responses:
        llm_id = response.llm_response.identifier
        evals = {}
        for eval_output in response.evaluations:
            if eval_output.evaluation_name in eval_names:
                evals[eval_output.evaluation_name] = {
                    f"evaluation_{eval_output.evaluation_name}_id": getattr(eval_output, "evaluation_id", None),
                    f"evaluation_{eval_output.evaluation_name}_outcome": getattr(eval_output, "evaluation_outcome", None)
                }
        row = {"llm_response_identifier": llm_id}
        for name in eval_names:
            row[f"evaluation_{name}_id"] = evals.get(name, {}).get(f"evaluation_{name}_id")
            row[f"evaluation_{name}_outcome"] = evals.get(name, {}).get(f"evaluation_{name}_outcome")
        rows[llm_id] = row
    return list(rows.values()), eval_names

def calculate_grounding_accuracy(merged_df):
    grounding_detectors = ["AWS_BEDROCK", "AZURE_CONTENT_SAFETY", "DEEPEVAL", "RAGAS"]
    filtered = merged_df[(merged_df["evaluation_abstention_outcome"] == "no") & ~(merged_df["retrieval_type"] == "DIRECT")].copy()
    print(f"Rows with evaluation_abstention_outcome == 'no' and retrieval_type != 'DIRECT': {len(filtered)}")
    for det in grounding_detectors:
        label_col = f"{det}_label"
        if label_col not in filtered:
            print(f"Detector {det} not found in columns.")
            continue
        det_rows = filtered[filtered[label_col].notna()]
        if len(filtered[filtered[label_col].isna()]) > 0:
            print(f"Some rows dropped due to NA labels for detector {det}.")
        unique_labels = set(det_rows[label_col].unique())
        if not unique_labels.issubset({"FACTUAL", "NON_FACTUAL"}):
            print(f"Warning: {det} has labels other than FACTUAL/NON_FACTUAL: {unique_labels}")
        percent_non_factual = (det_rows[label_col] == "NON_FACTUAL").mean() * 100
        print(f"Grounding accuracy for {det}: {percent_non_factual:.2f}% NON_FACTUAL ({len(det_rows)} rows)")

def calculate_grounding_similarity(merged_df):
    grounding_detectors = ["AWS_BEDROCK", "AZURE_CONTENT_SAFETY", "DEEPEVAL", "RAGAS"]
    filtered = merged_df[(merged_df["evaluation_abstention_outcome"] == "no") & ~(merged_df["retrieval_type"] == "DIRECT")].copy()
    print("\nGrounding similarity (pairwise label agreement):")
    label_cols = [f"{det}_label" for det in grounding_detectors if f"{det}_label" in filtered]
    for col1, col2 in itertools.combinations(label_cols, 2):
        valid = filtered[[col1, col2]].dropna()
        if valid.empty:
            print(f"No valid rows for {col1} vs {col2}")
            continue
        agreement = (valid[col1] == valid[col2]).mean() * 100
        print(f"{col1} vs {col2}: {agreement:.2f}% agreement over {len(valid)} rows")

def calculate_factual_accuracy(merged_df):
    print("\nFactual accuracy (GEMINI):")
    filtered = merged_df[merged_df["evaluation_abstention_outcome"] == "no"].copy()
    def overall_factuality_label(row):
        label = row.get("evaluation_factuality_outcome", None)
        if label is None:
            return None
        if str(label).lower() in ["tier_1", "tier_2"]:
            return "FACTUAL"
        return "NON_FACTUAL"
    filtered["overall_factuality_label"] = filtered.apply(overall_factuality_label, axis=1)
    gemini_col = "GEMINI_label"
    valid = filtered[filtered[gemini_col].notna() & filtered["overall_factuality_label"].notna()]
    match = (valid[gemini_col] == valid["overall_factuality_label"]).mean() * 100
    print(f"GEMINI factual accuracy: {match:.2f}% match with overall factuality label ({len(valid)} rows)")

    certain = valid[valid[gemini_col].notna() & valid["overall_factuality_label"].notna() & (valid["GEMINI_label"]!="UNCERTAIN")]
    match2 = (certain[gemini_col] == certain["overall_factuality_label"]).mean() * 100
    print(f"GEMINI factual accuracy for CERTAIN rows: {match2:.2f}% match with overall factuality label ({len(certain)} rows)")

def calculate_grounding_correlation_matrix(merged_df):
    grounding_detectors = ["AWS_BEDROCK", "AZURE_CONTENT_SAFETY", "DEEPEVAL", "RAGAS"]
    filtered = merged_df[(merged_df["evaluation_abstention_outcome"] == "no") & ~(merged_df["retrieval_type"] == "DIRECT")].copy()
    label_cols = [f"{det}_label" for det in grounding_detectors if f"{det}_label" in filtered]
    n = len(label_cols)
    matrix = [[None for _ in range(n)] for _ in range(n)]
    for i, col_i in enumerate(label_cols):
        col_i_non_factual = filtered[col_i] == "NON_FACTUAL"
        denom = col_i_non_factual.sum()
        for j, col_j in enumerate(label_cols):
            if denom == 0:
                matrix[i][j] = None
            else:
                num = ((filtered[col_i] == "NON_FACTUAL") & (filtered[col_j] == "NON_FACTUAL")).sum()
                matrix[i][j] = num / denom
    # Pretty print matrix with pandas DataFrame
    df_matrix = pd.DataFrame(matrix, index=label_cols, columns=label_cols)
    print("\nGrounding correlation matrix (P(det_j=NON_FACTUAL | det_i=NON_FACTUAL)):")
    print(df_matrix.round(2).to_string())

def get_subgroups(merged_df):
    """Get unique values for retrieval_type and knowledge_base_identifier separately."""
    retrieval_types = merged_df["retrieval_type"].dropna().unique().tolist()
    knowledge_bases = merged_df["knowledge_base_identifier"].dropna().unique().tolist()
    return {
        "retrieval_types": retrieval_types,
        "knowledge_bases": knowledge_bases
    }

def calculate_metrics_for_subgroup(df, group_type, group_value):
    """Calculate metrics for a specific subgroup."""
    print("\n" + "="*80)
    print(f"Metrics for {group_type}: {group_value}")
    print("="*80)
    calculate_grounding_accuracy(df)
    calculate_factual_accuracy(df)

def main():
    parser = argparse.ArgumentParser(description="Analyse detection and evaluation results and save as CSV.")
    parser.add_argument("--detected_dir", type=str, required=True, help="Path to the folder containing detection results.")
    parser.add_argument("--evaluated_dir", type=str, required=True, help="Path to the folder containing evaluation results.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the merged CSV.")
    args = parser.parse_args()

    detected_dir = Path(args.detected_dir)
    evaluated_dir = Path(args.evaluated_dir)

    # 1. Get all detector types
    detector_types = get_all_detector_types(detected_dir)
    print(f"Detector types found: {detector_types}")

    # 2. Load detection files and build detection dataframe
    detection_rows = extract_detection_rows(detected_dir, detector_types)
    detection_df = pd.DataFrame(detection_rows)
    print(f"Detection dataframe rows: {len(detection_df)}")

    # 3. Load evaluation files and build evaluation dataframe
    evaluation_rows = []
    all_eval_names = set()
    for file in evaluated_dir.glob("*.json"):
        try:
            doc = EvaluatedExperimentDocument.load_from_json(file)
            rows, eval_names = extract_evaluation_rows(doc)
            evaluation_rows.extend(rows)
            all_eval_names.update(eval_names)
        except Exception as e:
            print(f"Error loading evaluation file {file}: {e}")
    eval_columns = ["llm_response_identifier"]
    for name in sorted(all_eval_names):
        eval_columns.append(f"evaluation_{name}_id")
        eval_columns.append(f"evaluation_{name}_outcome")
    evaluation_df = pd.DataFrame(evaluation_rows).drop_duplicates("llm_response_identifier")
    evaluation_df = evaluation_df[eval_columns]
    print(f"Evaluation dataframe rows: {len(evaluation_df)}")

    # 4. Merge
    merged_df = pd.merge(detection_df, evaluation_df, on="llm_response_identifier", how="left")
    print(f"Merged dataframe rows: {len(merged_df)}")

    # 5. Save
    merged_df.to_csv(args.output_csv, index=False)
    print(f"Saved merged CSV to {args.output_csv}")

    # 6. Calculate metrics for all data
    print("\nCalculating metrics for all data:")
    calculate_grounding_accuracy(merged_df)
    calculate_factual_accuracy(merged_df)
    
    # 7. Calculate metrics for each subgroup type
    subgroups = get_subgroups(merged_df)
    
    # Calculate metrics by retrieval type
    print("\n" + "#"*80)
    print("Analyzing by Retrieval Type")
    print("#"*80)
    for retrieval_type in subgroups["retrieval_types"]:
        subgroup_df = merged_df[merged_df["retrieval_type"] == retrieval_type]
        if len(subgroup_df) > 0:
            calculate_metrics_for_subgroup(subgroup_df, "retrieval_type", retrieval_type)
        else:
            print(f"\nNo data for retrieval_type: {retrieval_type}")
    
    # Calculate metrics by knowledge base
    print("\n" + "#"*80)
    print("Analyzing by Knowledge Base")
    print("#"*80)
    for knowledge_base in subgroups["knowledge_bases"]:
        subgroup_df = merged_df[merged_df["knowledge_base_identifier"] == knowledge_base]
        if len(subgroup_df) > 0:
            calculate_metrics_for_subgroup(subgroup_df, "knowledge_base", knowledge_base)
        else:
            print(f"\nNo data for knowledge_base: {knowledge_base}")

    # 8. Calculate correlation matrix for all data
    calculate_grounding_correlation_matrix(merged_df)

if __name__ == "__main__":
    main()
