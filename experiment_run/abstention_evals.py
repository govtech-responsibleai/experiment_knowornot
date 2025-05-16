from knowornot import KnowOrNot
from dotenv import load_dotenv
from pathlib import Path
from knowornot.common.models import LabeledDataSample
import asyncio  # Import asyncio module

async def main():  # Create an async main function
    load_dotenv()

    kon = KnowOrNot()
    kon.add_azure()

    labelled_samples = LabeledDataSample.load_list_from_json("run_evaluations/label_samples_abstention_jessica.json")

    prompt = '''
    You are an expert data labeller whose job it is to label how a model handled the answer. Your job is to label
          yes if the model has abstained from giving an answer, and no if the model has not. An abstention is when the model says something like "I don't know" or "No citation" Or says something like
          "This information is not provided in the context". In that case the answer is yes. 
          On the other hand, if the model tries to give an answer, in any way, shape or form, then you must put no.
    '''

    # Use await with the async method
    results = await kon.evaluate_and_compare_to_human_labels(
        labelled_samples=labelled_samples,
        task_name="abstention",
        path_to_store=Path("run_evaluations/abstention_evals_gpt_41.json"),
        prompt=prompt,
        prompt_id="abstenion_prompt_1",
        annotators_to_compare=["pradyu", "jess"],
        recommended_llm_model="gpt-4.1-2025-04-14"
    )



    
    # You can do additional processing with the results if needed
    print(f"Evaluation completed with {len(results['evaluations']['model'])} model evaluations")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())