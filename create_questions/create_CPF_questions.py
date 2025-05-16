from pathlib import Path
from knowornot import KnowOrNot
from dotenv import load_dotenv

load_dotenv()
kon = KnowOrNot()
kon.add_azure()

kon.create_diverse_questions_from_QAPairs(
    knowledge_base_identifier="CPF",
    qa_pairs="data/CPF_raw_QA.json",
    method="both",
    path_to_save=Path("questions/CPF_filtered_qa.json"),
    diversity_threshold_semantic=.4,
    diversity_threshold_keyword=.4
)