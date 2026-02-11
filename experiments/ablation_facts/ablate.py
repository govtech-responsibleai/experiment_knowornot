from pathlib import Path
from knowornot import KnowOrNot
from dotenv import load_dotenv

import json, os

if __name__=="__main__":

    load_dotenv()
    kon = KnowOrNot()
    kon.add_azure()

    threshold = 0.5
    max_threshold = 0.54
    step = 0.1

    kb_directories = {
        # "CPF": Path("data/CPF_raw_QA.json"),
        # "ICA": Path('data/all_ICA_QA.json'),
        "BTT": list(Path("data/BTT_txts_from_pdf").glob("*.txt")),
        # "medishield": list(Path("data/InformationBooklet_pages").glob("*.txt")), 
    }


    for kb_name, source in kb_directories.items():

        # Ensure output directory exists
        output_dir = Path(f"experiments/ablation_facts/questions/{kb_name}")
        output_dir.mkdir(parents=True, exist_ok=True)


        while threshold <= max_threshold:  
            output_path = output_dir / f"{kb_name}_filtered_qa_sem{threshold:.2f}.json"
            
            if not os.path.exists(output_path): 
                if isinstance(source, Path) and source.suffix == ".json":
                    with open(source, "r", encoding="utf-8") as f:
                        qa_pairs = json.load(f)

                    kon.create_diverse_questions_from_QAPairs(
                        knowledge_base_identifier=kb_name,
                        qa_pairs=qa_pairs,
                        method="both",
                        path_to_save=output_path,
                        diversity_threshold_semantic=threshold,
                        diversity_threshold_keyword=0.4  # Fixed keyword threshold
                    )
                elif isinstance(source, list):
                    if kb_name == "BTT":
                        kon.create_questions(source_paths=source, knowledge_base_identifier=kb_name, 
                            context_prompt="This is a set of questions about Singapore's Basic Theory Test",
                            path_to_save_questions=output_path,
                            filter_method="both",
                            semantic_filter_threshold=threshold)
                            # keyword_filter_threshold=0.4) 
                    elif kb_name == "medishield":
                        kon.create_questions(source_paths=source, knowledge_base_identifier=kb_name, 
                            context_prompt="This is a set of data about Singapore's Medishield program",
                            path_to_save_questions=output_path,
                            filter_method="both",
                            semantic_filter_threshold=threshold,
                            keyword_filter_threshold=0.4) 
            threshold = round(threshold + step, 10)  # Avoid floating point precision issues