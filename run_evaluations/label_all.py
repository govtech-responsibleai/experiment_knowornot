from pathlib import Path
from knowornot import KnowOrNot
from knowornot.common.models import LabeledDataSample
from dotenv import load_dotenv
load_dotenv()
kon = KnowOrNot()
kon.add_azure()

samples = LabeledDataSample.load_list_from_json("run_evaluations/label_samples.json")
label_name = "abstention"
possible_values = ["yes", "no"]
allowed_inputs = ["question"]

kon.label_samples(labeled_samples=samples, label_name=label_name, possible_values=possible_values, allowed_inputs=allowed_inputs,
    path_to_save=Path("label_samples_abstention.json"))