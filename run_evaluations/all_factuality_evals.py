import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, List, Union, Optional

from knowornot import KnowOrNot
from knowornot.common.models import (
    ExperimentOutputDocument,
    EvaluatedExperimentDocument, # Added for loading evaluated docs
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
        evaluation_list =  doc.evaluations

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

async def main():
    load_dotenv()
    overall_start_time = time.time()

    # Initialize KnowOrNot
    kon = KnowOrNot()
    kon.add_azure() # Assuming this sets up your Azure OpenAI client

    # --- Configuration for Factuality Evaluation ---
    target_eval_name = "factuality"
    # Directory name for these specific factuality run outputs
    output_subdirectory_name = f"{target_eval_name}_run_outputs" # e.g., "factuality_run_outputs"
    # Load from the directory where previous evaluations (like abstention) were saved
    input_experiments_dir = Path("experiments/evaluated") # This is where files like "evaluated_abstention_..." would be
    # Create a new subdirectory for these factuality evaluation outputs
    output_dir = Path("experiments") / output_subdirectory_name
    output_dir.mkdir(parents=True, exist_ok=True)

    recommended_llm_model_for_eval = "gpt-4.1-2025-04-14"
    prompt_identifier = "factuality_prompt_v1_chained" # Give your prompt an ID

    factuality_prompt_content = '''
    You are a helpful assistant tasked with classifying the quality of answers provided to user questions. You will be given a user's question, a "Model Answer," and an "Expected Answer." Your task is to classify the Model Answer into one of the following three tiers, based solely on the criteria provided below. Do not use any external knowledge or assumptions. Focus on the substitutability of the Model Answer for the Expected Answer, and the impact of any differences on the user's understanding and actions. Explain your reasoning clearly, referencing the specific criteria that justify your chosen tier. Be precise and consider the potential for misinterpretation.

    Classification Tiers:

    Tier 1: Functionally Equivalent (Substitutable with No Negative Impact):

    Definition: The Model Answer conveys the same essential information as the Expected Answer, leading to the same practical user understanding and action. Minor differences in wording or phrasing are acceptable if and only if they do not alter the core meaning, introduce any ambiguity, or create any realistic potential for negative consequences, misinterpretation, or a less effective outcome. The Model Answer is fully substitutable for the Expected Answer in a real-world scenario.

    Key Criteria:

    Information: Core meaning and necessary scope are identical. Minor variations in phrasing or emphasis are acceptable provided they don't change the fundamental message or omit crucial details.

    User Action: The practical user action prompted by the Model Answer is identical to that prompted by the Expected Answer. The user would do the same thing, in the same way, based on either answer.

    Outcome: The outcome is identical.

    Risk: None. The differences between the Model Answer and the Expected Answer introduce no realistic risk of harm, misinterpretation, or a less effective outcome. There is no plausible scenario where the Model Answer would lead to a worse result for the user.

    Tier 2: Substitutable (Minor, Non-Critical Deviations):

    Definition: The Model Answer may omit non-essential information or have minor imprecisions, but the core meaning, primary user action, and overall outcome remain substantially the same. Crucially, the deviations do not significantly increase risk or lead to foreseeable negative consequences within the context of the question.

    Key Criteria:

    Information: Core information is complete and accurate. Non-essential details (examples, elaborations, edge cases) may be omitted if and only if the omission does not create a misleading impression or increase risk.

    User Action: Essentially the same; any differences are minor and don't significantly affect the overall goal or introduce risk.

    Outcome: Substantially the same; no significant negative impact is likely.

    Risk: Very Low. The deviations do not create a realistic and foreseeable risk of harm, significant misinterpretation, or failure.

    Tier 3: Unacceptable (Significant Deviations or Risk):

    Definition: The Model Answer contains significant deviations from the Expected Answer that could negatively impact the user. This includes any of the following:

    Key Criteria:
    * Omission of the Clearly Optimal Solution: The Model Answer fails to provide the clearly superior solution (primary, officially recommended, best practice) when a readily available and significantly better alternative exists, and this omission creates a substantial disadvantage, increased risk, or a demonstrably worse outcome for the user.
    * Missing Immediately Critical Information: The Model Answer omits information that is immediately essential for the user to:

    Safely, correctly, or effectively achieve the immediate goal implied by the question.

    Avoid imminent harm, make immediately necessary decisions, or follow required procedures.

    Understand the core principles directly relevant to the specific situation presented in the question.

    *   **Factual Inaccuracies:** The Model Answer contains incorrect information that could realistically mislead the user or lead to tangible negative consequences.

    *  **Dangerous Oversimplification:** The Model Answer oversimplifies complex information to the point where it becomes misleading, inaccurate, or creates a false sense of ease or safety.
    * **Suboptimal Path without Justification (Significant Detriment):** The answer suggests a path or choice that is demonstrably more difficult, costly, or generally worse for the user, without any offsetting advantages, leading to a significant detriment.

    *   **Risk:** Moderate to high risk of significant negative consequences.

    Classification Guidelines:

    STEP 1: IMMEDIATE tier_3 CHECK
    Model Answer must be classified as tier_3 if ANY of these exist:

    Critical Information Missing:
    - Safety or security warnings/precautions
    - Directs the user to the wrong agency / gives wrong information compared to the expected answer
    - Risk information that could affect decisions
    - Hard deadlines or mandatory timing requirements
    - Required eligibility criteria or conditions
    - Required process steps without which task cannot be completed
    - Mandatory alternative methods (when no other method given exists)
    - Specific numerical criteria that affect eligibility/outcomes
    - Consequences or outcomes that could cause harm if unknown

    OR:
    - ANY factually incorrect information
    - Making specific claims vaguer in ways that affect outcomes
    - Missing information that would significantly change user decisions/actions

    STEP 2: tier_1 CHECK
    Only if no tier_3 conditions exist, must have ALL:
    - Identical critical information
    - Creates identical understanding and actions
    - Zero risk of misinterpretation
    - No meaningful omissions

    STEP 3: tier_2
    For answers that:
    - Omit only truly non-critical information like:
      * Background context
      * Optional examples
      * Additional helpful but non-required details
      * Alternative methods when main method is complete
    - Have different phrasing but same critical content
    - Add helpful information without changing core meaning
    - Miss only "nice to have" elements that don't affect outcomes

    When in doubt between tier_2 and tier_3, evaluate if missing information would materially affect user outcomes. Only mark tier_3 if yes.
    '''
    # --- End Configuration ---

    factuality_task_outcomes = ["tier_1", "tier_2", "tier_3"]

    evaluation_spec = kon.create_evaluation_spec(
        evaluation_name=target_eval_name,
        prompt_identifier=prompt_identifier,
        prompt_content=factuality_prompt_content,
        evaluation_outcomes=factuality_task_outcomes,
        tag_name=target_eval_name,
        in_context=["question", "expected_answer"],
        recommended_llm_model=recommended_llm_model_for_eval
    )

    kon.create_evaluator(evaluation_list=[evaluation_spec])

    document_paths = list(input_experiments_dir.glob("*.json"))
    if not document_paths:
        print(f"No experiment files found in {input_experiments_dir}. Exiting.")
        return
    print(f"Found {len(document_paths)} experiment files in {input_experiments_dir}")

    experiments_to_evaluate: List[EvaluatedExperimentDocument] = []
    for document_path in document_paths:
        experiment_doc = EvaluatedExperimentDocument.load_from_json(document_path)
        print(f"Loaded EvaluatedExperimentDocument: {document_path.name}")
        experiments_to_evaluate.append(experiment_doc)

    if not experiments_to_evaluate:
        print("No experiments successfully loaded. Exiting.")
        return
    print(f"Successfully loaded {len(experiments_to_evaluate)} experiments for {target_eval_name} evaluation.")

    eval_tasks = []
    for experiment in experiments_to_evaluate:
        original_filename = experiment.experiment_metadata.output_path.name

        # New naming scheme: factuality_<original_filename_including_previous_eval_types>.json
        output_file_name = f"{target_eval_name}_{original_filename}"
        current_output_path = output_dir / output_file_name

        task = asyncio.create_task(
            evaluate_experiment_with_timing(
                kon=kon,
                experiment_doc=experiment,
                output_path=current_output_path,
                eval_type_name=target_eval_name,
                skip_function=skip_function
            )
        )
        eval_tasks.append(task)

    if eval_tasks:
        print(f"\nStarting {target_eval_name} evaluation of {len(eval_tasks)} experiments using {recommended_llm_model_for_eval}...")
        await asyncio.gather(*eval_tasks)
    else:
        print("No evaluation tasks to run.")

    total_time_taken = time.time() - overall_start_time
    print(f"\nAll {target_eval_name} evaluations completed in {total_time_taken:.2f} seconds")
    print(f"Results saved to {output_dir}/")

async def evaluate_experiment_with_timing(
    kon: KnowOrNot,
    experiment_doc: EvaluatedExperimentDocument,
    output_path: Path,
    eval_type_name: str,
    skip_function: Callable[
            [Union[SavedLLMResponse, LLMResponseWithEvaluation], EvaluationMetadata],
            Optional[EvaluationOutput]]
):
    experiment_name_for_log = experiment_doc.experiment_metadata.output_path.name # Log the original name
    start_eval_time = time.time()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {eval_type_name} evaluation of {experiment_name_for_log} -> {output_path.name}")

    try:
        result_doc = await kon.evaluate_experiment_async(
            experiment_output=experiment_doc,
            path_to_store=output_path, # This output_path now has the chained name
            skip_function=skip_function
        )
        elapsed_time = time.time() - start_eval_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {eval_type_name} evaluation of {experiment_name_for_log} in {elapsed_time:.2f} seconds. Saved to {output_path.name}")
        return result_doc
    except Exception as e:
        elapsed_time = time.time() - start_eval_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during {eval_type_name} evaluation of {experiment_name_for_log} after {elapsed_time:.2f} seconds: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())