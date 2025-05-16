from pathlib import Path
from knowornot import KnowOrNot
from dotenv import load_dotenv
import json
load_dotenv()

kon = KnowOrNot()
kon.add_azure()

with open('data/all_ICA_QA.json', 'r') as f:
    data = json.load(f)

questions = kon.create_diverse_questions_from_QAPairs(
    knowledge_base_identifier="ICA",
    qa_pairs=data,
    method="both",
    path_to_save=Path("questions/ICA_filtered_qa.json")
)

