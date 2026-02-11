from pathlib import Path
from knowornot import KnowOrNot
from dotenv import load_dotenv
from knowornot.SyntheticExperimentCreator import SyntheticExperimentCreator
from knowornot.SyncLLMClient import SyncLLMClientEnum
from knowornot.common.models import QAPair
import pandas as pd
import json, os
import logging
logger = logging.getLogger("SyntheticQA")

if __name__=="__main__":

    load_dotenv()
    kon = KnowOrNot()
    kon.add_gemini()
    model = 'gemini'
    

    kb_directories = {
        # "ICA": Path('data/all_ICA_QA.json'),
        # "CPF": Path("data/CPF_raw_QA.json"),
        "BTT": list(Path("data/BTT_txts_from_pdf").glob("*.txt")),
        "medishield": list(Path("data/InformationBooklet_pages").glob("*.txt")), 
    }

    for kb_name, source in kb_directories.items():

        # Ensure output directory exists
        output_dir = Path(f"experiments/ablation_facts/questions/{kb_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{kb_name}_synthetic_qa_{model}.json"

        if not os.path.exists(output_path): 
            if kb_name in ["CPF", "ICA"]:
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                qa_pairs = [QAPair(question=item["question"], answer=item["answer"]) for item in data if item["status"] == "success"]
            elif kb_name in ["BTT", "medishield"]:
                intermediate_path = output_dir / f"{kb_name}_synthetic_qa_{model}_intermediate.json"
                if not intermediate_path.exists():
                    kon.create_questions(source_paths=source, knowledge_base_identifier=kb_name, 
                        context_prompt="This is a set of data about Singapore's Medishield program" if kb_name=="medishield" else "This is a set of questions about Singapore's Basic Theory Test",
                        path_to_save_questions=intermediate_path,
                        filter_method="keyword",
                        keyword_filter_threshold=0.0) 
                with open(intermediate_path,"r", encoding="utf-8") as f:
                    data = json.load(f) 
                qa_pairs = [QAPair(question=item["question"], answer=item["answer"]) for item in data["questions"]]
                print(f"Generated intermediate, fact-based QA pairs for {kb_name}.")

            if kb_name in ["CPF", "ICA"]:
                perc = 0.5
            elif kb_name == "BTT":
                perc = 0.1
            elif kb_name == "medishield":
                perc = 0.7
        
            if model == "openai":
                client_name = SyncLLMClientEnum.AZURE_OPENAI
            elif model == "gemini":
                client_name = SyncLLMClientEnum.GEMINI
            elif model == "claude":
                client_name = SyncLLMClientEnum.ANTHROPIC
            llm_client =  kon.get_client(client_name)
            experiment_creator = SyntheticExperimentCreator(
                default_client=llm_client,
                default_synthetic_prompt="You will be given questions in a cluster. Your task is to generate new questions that are related to the topic, but are different and distinct. Ensure that they are not informationally or semantically the same as any questions in the cluster.",
                default_synthetic_check_prompt="",
                logger=logger,
                default_percentage=perc,  # Generate 50% as many new questions as originals per cluster
            )

            # Step 6: Generate synthetic dataset
            synthetic_qas, clusters = experiment_creator.generate_synthetic_dataset(
                qa_pair_list=qa_pairs,
                num_clusters=1,  # You can tune this
                percentage=perc,  # Optional override
            )

            print(synthetic_qas)
                            # Step 7: Save the synthetic QA pairs
            synthetic_data = [
                {"question": qa.question} for qa in synthetic_qas
            ]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(synthetic_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(synthetic_qas)} synthetic QA pairs to {output_path}")
       
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        df.to_csv(output_dir / f"{kb_name}_synthetic_qa_{model}.csv", index=False, encoding="utf-8")


            

            
