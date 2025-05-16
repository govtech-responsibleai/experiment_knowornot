from pathlib import Path
from knowornot import KnowOrNot
from dotenv import load_dotenv

load_dotenv()
kon = KnowOrNot()
kon.add_azure()

input_data_dir = Path("data/BTT_txts_from_pdf")
input_data_text = list(input_data_dir.glob("*.txt"))

create_questions = kon.create_questions(source_paths=input_data_text, knowledge_base_identifier="BTT", 
    context_prompt="This is a set of questions about Singapore's Basic Theory Test",
    path_to_save_questions=Path("questions/BTT_QA.json"),
    filter_method="both")

