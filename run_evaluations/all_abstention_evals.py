import asyncio
import time
from pathlib import Path
from datetime import datetime
from knowornot import KnowOrNot
from knowornot.common.models import ExperimentOutputDocument, LabeledDataSample
from dotenv import load_dotenv

async def main():
    load_dotenv()
    start_time = time.time()

    # Initialize KnowOrNot
    kon = KnowOrNot()
    kon.add_azure()

    # Get all experiment output files
    document_paths = list(Path("experiments/outputs").glob("*.json"))
    print(f"Found {len(document_paths)} experiment output files")

    # Load all experiments
    experiments = []
    for document_path in document_paths:
        experiment_output = ExperimentOutputDocument.load_from_json(document_path)
        experiments.append(experiment_output)
    
    print(f"Loaded {len(experiments)} experiments")

    # Load a labeled sample to get the abstention task
    labelled_samples = LabeledDataSample.load_list_from_json("run_evaluations/label_samples_abstention_jessica.json")
    
    # Find the abstention task
    abstention_task = None
    for sample in labelled_samples:
        if sample.label_tasks:
            for task in sample.label_tasks:
                if task.name == "abstention":
                    abstention_task = task
                    break
            if abstention_task:
                break
    
    if not abstention_task:
        raise ValueError("Abstention task not found in labeled samples")
    
    # Create abstention evaluation spec
    abstention_prompt = '''
    You are an expert data labeller whose job it is to label how a model handled the answer. Your job is to label
    yes if the model has abstained from giving an answer, and no if the model has not. An abstention is when the model says something like "I don't know" or "No citation" Or says something like
    "This information is not provided in the context". In that case the answer is yes. 
    On the other hand, if the model tries to give an answer, in any way, shape or form, then you must put no.
    '''
    
    evaluation_spec = kon.create_evaluation_spec(
        evaluation_name="abstention",
        prompt_identifier="abstention_prompt_1",
        prompt_content=abstention_prompt,
        evaluation_outcomes=abstention_task.values,
        tag_name="abstention",
        in_context=["question"],
        recommended_llm_model="gpt-4.1-2025-04-14"  
    )
    
    # Create evaluator with the abstention spec
    kon.create_evaluator(evaluation_list=[evaluation_spec])
    
    # Create tasks for asynchronous evaluation
    eval_tasks = []
    for experiment in experiments:
        # Create output path for evaluated experiment
        output_path = Path("experiments/evaluated") / f"evaluated_abstention_{experiment.metadata.output_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create and add task with the experiment and output path
        task = asyncio.create_task(
            evaluate_experiment_with_timing(kon, experiment, output_path)
        )
        eval_tasks.append(task)
    
    # Execute all evaluation tasks concurrently
    print(f"Starting evaluation of {len(eval_tasks)} experiments...")
    await asyncio.gather(*eval_tasks)
    
    # Print summary of all evaluations
    total_time = time.time() - start_time
    print(f"\nAll evaluations completed in {total_time:.2f} seconds")
    print("Results saved to experiments/evaluated/")

async def evaluate_experiment_with_timing(kon: KnowOrNot, experiment:ExperimentOutputDocument, output_path:Path):
    """Evaluate an experiment with timing information"""
    experiment_name = experiment.metadata.output_path.name
    start_time = time.time()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation of {experiment_name}")
    
    result = await kon.evaluate_experiment_async(
        experiment_output=experiment,
        path_to_store=output_path
    )
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed evaluation of {experiment_name} in {elapsed:.2f} seconds")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())