from knowornot import KnowOrNot
from knowornot.common.models import QuestionDocument

kon = KnowOrNot()
kon.add_azure()

btt_question_doc = QuestionDocument.load_from_json("questions/BTT_QA.json")
ica_question_doc = QuestionDocument.load_from_json("questions/ICA_filtered_qa.json")
cpf_question_doc = QuestionDocument.load_from_json("questions/CPF_filtered_qa.json")
medishield_question_doc = QuestionDocument.load_from_json("questions/medishield_QA.json")


kon.create_all_inputs_for_experiment(question_document=btt_question_doc)

kon.create_all_inputs_for_experiment(question_document=ica_question_doc)

kon.create_all_inputs_for_experiment(question_document=cpf_question_doc)

kon.create_all_inputs_for_experiment(question_document=medishield_question_doc)