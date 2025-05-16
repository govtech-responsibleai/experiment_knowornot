from pathlib import Path
from knowornot import KnowOrNot
from knowornot.common.models import LabeledDataSample
kon = KnowOrNot()

samples = LabeledDataSample.load_list_from_json("label_samples_factuality.json")
label_name = "factuality"
possible_values = ["tier_1", "tier_2", "tier_3"]
allowed_inputs = ["question", "expected_answer"]

kon.label_samples(labeled_samples=samples, label_name=label_name, possible_values=possible_values, allowed_inputs=allowed_inputs,
    path_to_save=Path("run_evaluations/label_samples_factuality_final.json"))