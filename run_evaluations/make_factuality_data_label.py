from typing import Union
from knowornot import KnowOrNot
from knowornot.common.models import EvaluatedExperimentDocument, LLMResponseWithEvaluation, SavedLLMResponse
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

evaluated_experiment_paths = Path("experiments/evaluated").glob("*.json")

kon = KnowOrNot()
kon.add_azure()

evaluated_experiments = []
for evaluated_experiment_path in evaluated_experiment_paths:
    evaluated_experiment = EvaluatedExperimentDocument.load_from_json(evaluated_experiment_path)
    evaluated_experiments.append(evaluated_experiment)


def filter_abstention(response: Union[LLMResponseWithEvaluation, SavedLLMResponse]) -> bool:
    if isinstance(response, SavedLLMResponse):
        return False 
    

    evaluations = response.evaluations
    for evaluation in evaluations:
        if evaluation.evaluation_name == "abstention" and evaluation.evaluation_outcome == "no":
            return True
    return False

samples = kon.create_samples_to_label(experiment_outputs=evaluated_experiments, path_to_store=Path("run_evaluations/label_samples_factuality.json"), percentage_to_sample=.1, 
                                      filter_function=filter_abstention)