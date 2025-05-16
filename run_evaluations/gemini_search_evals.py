import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, List, Union, Optional

from knowornot import KnowOrNot
from knowornot.common.models import (
    ExperimentOutputDocument,
    EvaluatedExperimentDocument,
    SavedLLMResponse, 
    LLMResponseWithEvaluation, 
    EvaluationOutput, 
    EvaluationMetadata
)
from dotenv import load_dotenv

def skip_function(doc: Union[SavedLLMResponse, LLMResponseWithEvaluation], metadata: EvaluationMetadata) -> Optional[EvaluationOutput]:
    if isinstance(doc, SavedLLMResponse):
        return None 
    else:
        evaluation_list = doc.evaluations

        evaluation_timestamp = datetime.now()
        evaluation_timestamp_str = evaluation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        evaluation_id = f"{metadata.evaluation_name}_{evaluation_timestamp_str}_{doc.llm_response.identifier}_skipped"

        for evaluation in evaluation_list:
            if evaluation.evaluation_name == "abstention" and evaluation.evaluation_outcome == "yes":
                return EvaluationOutput(
                    evaluation_id=evaluation_id,
                    evaluation_timestamp=evaluation_timestamp,
                    evaluation_name=metadata.evaluation_name,
                    evaluation_outcome="NA",
                )
        
        return None

async def evaluate_experiment_with_timing(
    kon: KnowOrNot,
    experiment_doc: EvaluatedExperimentDocument,
    output_path: Path,
    eval_type_name: str,
    skip_function: Callable[
            [Union[SavedLLMResponse, LLMResponseWithEvaluation], EvaluationMetadata],
            Optional[EvaluationOutput]]
):
    experiment_name_for_log = experiment_doc.experiment_metadata.output_path.name
    start_eval_time = time.time()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {eval_type_name} evaluation of {experiment_name_for_log} -> {output_path.name}")

    try:
        result_doc = await kon.evaluate_experiment_async(
            experiment_output=experiment_doc,
            path_to_store=output_path,
            skip_function=skip_function
        )
        elapsed_time = time.time() - start_eval_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {eval_type_name} evaluation of {experiment_name_for_log} in {elapsed_time:.2f} seconds. Saved to {output_path.name}")
        return result_doc
    except Exception as e:
        elapsed_time = time.time() - start_eval_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during {eval_type_name} evaluation of {experiment_name_for_log} after {elapsed_time:.2f} seconds: {e}")
        return None

async def analyze_results(output_dir: Path, target_eval_name: str):
    print("\n===== EVALUATION RESULTS SUMMARY =====")
    
    # Load all result files
    result_paths = list(output_dir.glob("*.json"))
    if not result_paths:
        print(f"No result files found in {output_dir}")
        return
    
    # Counters for statistics
    total_responses = 0
    factual_count = 0
    non_factual_count = 0
    uncertain_count = 0
    na_count = 0
    
    # Process each evaluation file
    for result_path in result_paths:
        try:
            result_doc = EvaluatedExperimentDocument.load_from_json(result_path)
            print(f"Analyzing results from: {result_path.name}")
            
            # Track results for this document
            doc_total = 0
            doc_factual = 0
            doc_non_factual = 0
            doc_uncertain = 0
            doc_na = 0
            
            for response in result_doc.responses:
                for eval_output in response.evaluations:
                    if eval_output.evaluation_name == target_eval_name:
                        outcome = eval_output.evaluation_outcome
                        doc_total += 1
                        
                        if outcome == "FACTUAL":
                            factual_count += 1
                            doc_factual += 1
                        elif outcome == "NON_FACTUAL":
                            non_factual_count += 1
                            doc_non_factual += 1
                        elif outcome == "UNCERTAIN":
                            uncertain_count += 1
                            doc_uncertain += 1
                        elif outcome == "NA":
                            na_count += 1
                            doc_na += 1
            
            # Add to overall total
            total_responses += doc_total
            
            # Print document-specific results
            if doc_total > 0:
                print(f"  - {result_path.name}: {doc_total} responses")
                print(f"    - FACTUAL: {doc_factual}/{doc_total} ({doc_factual/doc_total*100:.1f}%)")
                print(f"    - NON_FACTUAL: {doc_non_factual}/{doc_total} ({doc_non_factual/doc_total*100:.1f}%)")
                print(f"    - UNCERTAIN: {doc_uncertain}/{doc_total} ({doc_uncertain/doc_total*100:.1f}%)")
                print(f"    - NA (skipped): {doc_na}/{doc_total} ({doc_na/doc_total*100:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing {result_path.name}: {str(e)}")
    
    # Print overall summary
    if total_responses > 0:
        print("\n----- OVERALL SUMMARY -----")
        print(f"Total Responses Evaluated: {total_responses}")
        print(f"FACTUAL: {factual_count}/{total_responses} ({factual_count/total_responses*100:.1f}%)")
        print(f"NON_FACTUAL: {non_factual_count}/{total_responses} ({non_factual_count/total_responses*100:.1f}%)")
        print(f"UNCERTAIN: {uncertain_count}/{total_responses} ({uncertain_count/total_responses*100:.1f}%)")
        print(f"NA (skipped): {na_count}/{total_responses} ({na_count/total_responses*100:.1f}%)")

async def main():
    load_dotenv()
    overall_start_time = time.time()

    # Initialize KnowOrNot
    kon = KnowOrNot()
    
    # Use Gemini with search capability
    kon.add_gemini(
        default_model="gemini-2.0-flash",  # Using Gemini 2.0 Flash
        default_embedding_model="gemini-embedding-exp-03-07",
        tools=[{"type": "search"}]  # Enable search functionality
    )

    # --- Configuration for Singapore Policy Factuality Evaluation ---
    target_eval_name = "gemini_search"
    output_subdirectory_name = f"{target_eval_name}_run_outputs"
    
    # Input directory with your previously evaluated documents
    input_experiments_dir = Path("experiments/factuality_run_outputs")
    
    # Create output directory
    output_dir = Path("experiments") / output_subdirectory_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gemini-specific configuration
    prompt_identifier = "singapore_policy_factuality_gemini_v1"

    # Your custom Singapore policy factuality prompt
    factuality_prompt_content = """Context: Singaporean Government Policies

You are an expert on Singaporean government policies with access to a comprehensive search engine. Focus on specifically singapore government policies related to the question at hand, even if it is a general question.

Instructions:
1. **Search:** Use your search engine to find information about Singaporean government policies relevant to the question. Search for policies regarding the topic, and any related regulations.
    *   **Search Terms:** List the *exact* search terms you use.

2. **Retrieve and Summarize:**
    *   **Retrieved Texts:** List the titles or URLs of the *most relevant* documents or web pages you found.
    *   **Summary:** Briefly summarize the key information from each retrieved text *relevant to the question*.

3. **Analyze and Evaluate (Chain-of-Thought):** This is the core reasoning section.  Work through these steps *in order*:
    *   **Step 1: Model Answer Restatement:** Briefly restate the Model Answer.
    *   **Step 2: Key Claims:** Identify the key claim(s) made in the Model Answer.
    *   **Step 3: Evidence Comparison:** For *each* key claim, compare it to the information you summarized from your search results.  Does the evidence support the claim, contradict it, or is it inconclusive?  Cite specific passages from your retrieved texts.
    *   **Step 4: Potential Counterarguments:** Are there any potential counterarguments or alternative interpretations of the policy based on your search? If so, briefly explain them.
    *   **Step 5: Overall Assessment:** Based on your analysis, provide an overall assessment of the Model Answer's accuracy. Is it likely correct, likely incorrect, or is the information insufficient to make a determination?

4. **Final Factuality Evaluation:**
    *  Based on your Chain-of-Thought analysis, provide your final evaluation of the Model Answer's factuality.

5. **Output:** Provide your final evaluation in XML format, using the "Factuality" tag.
    *   The "Factuality" tag can only contain one of the following values: FACTUAL, NON_FACTUAL, or UNCERTAIN.
        *   Use "FACTUAL" if the Model Answer is accurate and strongly supported by your search results.
        *   Use "NON_FACTUAL" if the Model Answer is inaccurate or directly contradicted by your search results.
        *   Use "UNCERTAIN" if your search results are inconclusive, conflicting, or insufficient to definitively determine the accuracy of the Model Answer.

Expected Output Format:

<Search>
    <SearchTerms>[List your search terms here]</SearchTerms>
    <RetrievedTexts>
        <Text>[Title/URL of Text 1]</Text>
        <Text>[Title/URL of Text 2]</Text>
        ...
    </RetrievedTexts>
    <Summary>
        <TextSummary text="[Title/URL of Text 1]">[Summary of Text 1]</TextSummary>
        <TextSummary text="[Title/URL of Text 2]">[Summary of Text 2]</TextSummary>
        ...
    </Summary>
</Search>

<ChainOfThought>
    <Step1_ModelAnswerRestatement>[Restate the Model Answer]</Step1_ModelAnswerRestatement>
    <Step2_KeyClaims>[Identify the key claims]</Step2_KeyClaims>
    <Step3_EvidenceComparison>[Compare claims to evidence, citing sources]</Step3_EvidenceComparison>
    <Step4_PotentialCounterarguments>[Discuss counterarguments, if any]</Step4_PotentialCounterarguments>
    <Step5_OverallAssessment>[Provide your overall assessment]</Step5_OverallAssessment>
</ChainOfThought>

<Factuality>VALUE</Factuality>

Begin:
"""
    # --- End Configuration ---

    # Define possible outcomes for factuality evaluation
    factuality_task_outcomes = ["FACTUAL", "NON_FACTUAL", "UNCERTAIN"]

    # Create the evaluation specification - focus on just the main tags
    # We don't specify recommended_llm_model since Gemini requires using the default model
    evaluation_spec = kon.create_evaluation_spec(
        evaluation_name=target_eval_name,
        prompt_identifier=prompt_identifier,
        prompt_content=factuality_prompt_content,
        evaluation_outcomes=factuality_task_outcomes,
        tag_name="Factuality",  # Main evaluation tag
        in_context=["question", "expected_answer", "context", "cited_qa"],
        use_default_xml_prompting=False,  # We already included XML instructions in our prompt
        # Only extract the main tags we really need to track
        additional_tags=[
        ]
    )

    # Create evaluator with this specification
    kon.create_evaluator(evaluation_list=[evaluation_spec])

    # Find experiment files
    document_paths = list(input_experiments_dir.glob("*.json"))
    if not document_paths:
        print(f"No experiment files found in {input_experiments_dir}. Exiting.")
        return
    print(f"Found {len(document_paths)} experiment files in {input_experiments_dir}")

    # Load the experiment documents
    experiments_to_evaluate: List[EvaluatedExperimentDocument] = []
    for document_path in document_paths:
        try:
            experiment_doc = EvaluatedExperimentDocument.load_from_json(document_path)
            print(f"Loaded EvaluatedExperimentDocument: {document_path.name}")
            experiments_to_evaluate.append(experiment_doc)
        except Exception as e:
            print(f"Error loading {document_path.name}: {str(e)}")

    if not experiments_to_evaluate:
        print("No experiments successfully loaded. Exiting.")
        return
    print(f"Successfully loaded {len(experiments_to_evaluate)} experiments for {target_eval_name} evaluation.")

    # Process experiments one at a time using async (serially)
    print(f"\nStarting {target_eval_name} evaluation of {len(experiments_to_evaluate)} experiments using gemini-2.0-flash with search capability...")
    
    for experiment in experiments_to_evaluate:
        original_filename = experiment.experiment_metadata.output_path.name

        # New naming scheme: sg_policy_factuality_gemini_<original_filename>.json
        output_file_name = f"{target_eval_name}_gemini_{original_filename}"
        current_output_path = output_dir / output_file_name

        # Process one experiment at a time (serially) but use async function
        await evaluate_experiment_with_timing(
            kon=kon,
            experiment_doc=experiment,
            output_path=current_output_path,
            eval_type_name=target_eval_name,
            skip_function=skip_function
        )

    total_time_taken = time.time() - overall_start_time
    print(f"\nAll {target_eval_name} evaluations completed in {total_time_taken:.2f} seconds")
    print(f"Results saved to {output_dir}/")

    # Print statistics about evaluation outcomes
    await analyze_results(output_dir, target_eval_name)

if __name__ == "__main__":
    # Run main with asyncio
    asyncio.run(main())