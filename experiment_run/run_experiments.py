from pathlib import Path
from knowornot import KnowOrNot
from knowornot.common.models import ExperimentInputDocument
from dotenv import load_dotenv
import asyncio
import time

async def run_experiments_one_by_one():
    load_dotenv()

    kon = KnowOrNot()
    kon.add_azure()

    experiment_input_path_list = list(Path("experiments/inputs").glob("*.json"))
    
    results = []
    for i, experiment_input_path in enumerate(experiment_input_path_list):
        
        print(f"Running experiment {i+1}/{len(experiment_input_path_list)}: {experiment_input_path.name}")
        
        # Load the experiment input
        experiment_input = ExperimentInputDocument.load_from_json(experiment_input_path)

        if experiment_input.metadata.output_path.exists():
            print(f"Experiment {experiment_input_path.name} already completed")
            continue
        
        # Measure start time
        start_time = time.time()
        
        # Run the experiment asynchronously
        result = await kon.run_experiment_async(experiment_input)
        
        # Calculate duration
        duration = time.time() - start_time
        
        print(f"Completed experiment {experiment_input_path.name} in {duration:.2f} seconds")
        results.append(result)
    
    print(f"All {len(results)} experiments completed")
    return results

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_experiments_one_by_one())