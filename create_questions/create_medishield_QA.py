from knowornot import KnowOrNot
from knowornot.common.models import QuestionDocument
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

kon = KnowOrNot()
kon.add_azure()

input_files = list(Path("data/InformationBooklet_pages").glob("*.txt"))

question_document: QuestionDocument = kon.create_questions(source_paths=input_files, knowledge_base_identifier="medishield", 
                     context_prompt="This is a set of data about Singapore's Medishield program",
                     path_to_save_questions=Path("questions/medishield_QA.json"),
                     filter_method="both")

question_document.save_to_json()