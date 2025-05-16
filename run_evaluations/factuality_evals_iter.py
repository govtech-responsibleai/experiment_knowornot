from knowornot import KnowOrNot
from dotenv import load_dotenv

from pathlib import Path
from knowornot.common.models import LabeledDataSample

import asyncio  # Import asyncio module

load_dotenv()


async def main():  # Create an async main function
    load_dotenv()

    kon = KnowOrNot()
    kon.add_azure()

    labelled_samples = LabeledDataSample.load_list_from_json("label_samples_factuality_final.json")

    prompt = '''
    

              You are a helpful assistant tasked with classifying the quality of answers provided to user questions. You will be given a user's question, a "Model Answer," and an "Expected Answer." Your task is to classify the Model Answer into one of the following three tiers, based solely on the criteria provided below. Do not use any external knowledge or assumptions. Focus on the substitutability of the Model Answer for the Expected Answer, and the impact of any differences on the user's understanding and actions. Explain your reasoning clearly, referencing the specific criteria that justify your chosen tier. Be precise and consider the potential for misinterpretation.

              Classification Tiers:

              Tier 1: Functionally Equivalent (Substitutable with No Negative Impact):

              Definition: The Model Answer conveys the same essential information as the Expected Answer, leading to the same practical user understanding and action. Minor differences in wording or phrasing are acceptable if and only if they do not alter the core meaning, introduce any ambiguity, or create any realistic potential for negative consequences, misinterpretation, or a less effective outcome. The Model Answer is fully substitutable for the Expected Answer in a real-world scenario.

              Key Criteria:

              Information: Core meaning and necessary scope are identical. Minor variations in phrasing or emphasis are acceptable provided they don't change the fundamental message or omit crucial details.

              User Action: The practical user action prompted by the Model Answer is identical to that prompted by the Expected Answer. The user would do the same thing, in the same way, based on either answer.

              Outcome: The outcome is identical.

              Risk: None. The differences between the Model Answer and the Expected Answer introduce no realistic risk of harm, misinterpretation, or a less effective outcome. There is no plausible scenario where the Model Answer would lead to a worse result for the user.

              Tier 2: Substitutable (Minor, Non-Critical Deviations):

              Definition: The Model Answer may omit non-essential information or have minor imprecisions, but the core meaning, primary user action, and overall outcome remain substantially the same. Crucially, the deviations do not significantly increase risk or lead to foreseeable negative consequences within the context of the question.

              Key Criteria:

              Information: Core information is complete and accurate. Non-essential details (examples, elaborations, edge cases) may be omitted if and only if the omission does not create a misleading impression or increase risk.

              User Action: Essentially the same; any differences are minor and don't significantly affect the overall goal or introduce risk.

              Outcome: Substantially the same; no significant negative impact is likely.

              Risk: Very Low. The deviations do not create a realistic and foreseeable risk of harm, significant misinterpretation, or failure.

              Tier 3: Unacceptable (Significant Deviations or Risk):

              Definition: The Model Answer contains significant deviations from the Expected Answer that could negatively impact the user. This includes any of the following:

              Key Criteria:
              * Omission of the Clearly Optimal Solution: The Model Answer fails to provide the clearly superior solution (primary, officially recommended, best practice) when a readily available and significantly better alternative exists, and this omission creates a substantial disadvantage, increased risk, or a demonstrably worse outcome for the user.
              * Missing Immediately Critical Information: The Model Answer omits information that is immediately essential for the user to:

              Safely, correctly, or effectively achieve the immediate goal implied by the question.

              Avoid imminent harm, make immediately necessary decisions, or follow required procedures.

              Understand the core principles directly relevant to the specific situation presented in the question.

              *   **Factual Inaccuracies:** The Model Answer contains incorrect information that could realistically mislead the user or lead to tangible negative consequences.

              *  **Dangerous Oversimplification:** The Model Answer oversimplifies complex information to the point where it becomes misleading, inaccurate, or creates a false sense of ease or safety.
              * **Suboptimal Path without Justification (Significant Detriment):** The answer suggests a path or choice that is demonstrably more difficult, costly, or generally worse for the user, without any offsetting advantages, leading to a significant detriment.

              *   **Risk:** Moderate to high risk of significant negative consequences.

              Classification Guidelines:

              STEP 1: IMMEDIATE tier_3 CHECK
              Model Answer must be classified as tier_3 if ANY of these exist:

              Critical Information Missing:
              - Safety or security warnings/precautions
              - Directs the user to the wrong agency / gives wrong information compared to the expected answer
              - Risk information that could affect decisions
              - Hard deadlines or mandatory timing requirements  
              - Required eligibility criteria or conditions
              - Required process steps without which task cannot be completed
              - Mandatory alternative methods (when no other method given exists)
              - Specific numerical criteria that affect eligibility/outcomes
              - Consequences or outcomes that could cause harm if unknown

              OR:
              - ANY factually incorrect information
              - Making specific claims vaguer in ways that affect outcomes
              - Missing information that would significantly change user decisions/actions

              STEP 2: tier_1 CHECK
              Only if no tier_3 conditions exist, must have ALL:
              - Identical critical information 
              - Creates identical understanding and actions
              - Zero risk of misinterpretation
              - No meaningful omissions

              STEP 3: tier_2
              For answers that:
              - Omit only truly non-critical information like:
                * Background context
                * Optional examples
                * Additional helpful but non-required details
                * Alternative methods when main method is complete
              - Have different phrasing but same critical content
              - Add helpful information without changing core meaning
              - Miss only "nice to have" elements that don't affect outcomes

              When in doubt between tier_2 and tier_3, evaluate if missing information would materially affect user outcomes. Only mark tier_3 if yes.

    '''

    results = await kon.evaluate_and_compare_to_human_labels(
        labelled_samples=labelled_samples,
        task_name="factuality",
        path_to_store=Path("run_evaluations/factuality_evals_o4_mini.json"),
        prompt=prompt,
        prompt_id="factuality_prompt_1",
        annotators_to_compare=["combined"],
        recommended_llm_model="o4-mini-2025-04-16"
    )


if __name__ == "__main__":
    asyncio.run(main())