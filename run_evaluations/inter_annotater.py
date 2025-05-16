from knowornot import KnowOrNot
from knowornot.common.models import LabeledDataSample

kon = KnowOrNot()

labelled_samples = LabeledDataSample.load_list_from_json("run_evaluations/label_samples_abstention_jessica.json")
kon.find_inter_annotator_reliability(labeled_samples=labelled_samples, task_name="abstention")