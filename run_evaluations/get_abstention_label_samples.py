from pathlib import Path
from knowornot import KnowOrNot
from knowornot.common.models import ExperimentOutputDocument

from dotenv import load_dotenv
load_dotenv()

kon = KnowOrNot()
kon.add_azure()

experiment_files_path = Path("experiments/outputs")

experiment_files = list(experiment_files_path.glob("*.json"))

experiments = []
for experiment_file in experiment_files:
    experiment_output = ExperimentOutputDocument.load_from_json(experiment_file)
    experiments.append(experiment_output)

samples = kon.create_samples_to_label(experiment_outputs=experiments, path_to_store=Path("run_evaluations/label_samples.json"), percentage_to_sample=.1)

