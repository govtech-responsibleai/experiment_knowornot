from pathlib import Path
from knowornot import KnowOrNot
from knowornot.Detector.models import DetectorType, DetectionResult, DetectionStatus, DetectedExperimentDocument
import argparse
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize KnowOrNot
kon = KnowOrNot()

# Register TLM detector with your API key
kon.register_detector(
    detector_type=DetectorType.TLM,
    config={"api_key": os.getenv("TLM_API_KEY")}
)

# Register AWS Bedrock detector with your AWS credentials
kon.register_detector(
    detector_type=DetectorType.AWS_BEDROCK,
    config={
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION"),
        "guardrail_identifier": os.getenv("AWS_GUARDRAIL_IDENTIFIER"),
        "guardrail_version": os.getenv("AWS_GUARDRAIL_VERSION"),
    }
)

# Register the Azure Content Safety detector
kon.register_detector(
    detector_type=DetectorType.AZURE_CONTENT_SAFETY,
    config={
        "api_key": os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        "endpoint": os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        "region": os.getenv("AZURE_CONTENT_SAFETY_REGION"),  
        "api_version": os.getenv("AZURE_CONTENT_SAFETY_API_VERSION")
    }
)

kon.register_detector(
    detector_type=DetectorType.DEEPEVAL,
    config={
        "openai_api_key": os.getenv("DEEPEVAL_OPENAI_KEY")
    }
)

kon.register_detector(
    detector_type=DetectorType.RAGAS,
    config={
        "azure_api_key": os.getenv("AZURE_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_API_ENDPOINT"),
        "azure_api_version": os.getenv("AZURE_API_VERSION"), 
        "azure_model": os.getenv("AZURE_API_MODEL"),            
        "azure_deployment": os.getenv("AZURE_API_DEPLOYMENT"),
    }
)


# Define a skip function to skip detections for responses where evaluation outcome is "no"
def skip_non_abstained_responses(response, detector_type):
    # Check if there are any evaluations
    if not response.evaluations:
        return None  # Run detection
    
    # If any evaluation has outcome "yes", skip detection
    if any(eval.evaluation_outcome == "yes" and eval.evaluation_name == "abstention" for eval in response.evaluations):
        return DetectionResult(
            detector_type=detector_type,
            confidence=None,
            explanation="Skipped: Response was abstained",
            status=DetectionStatus.SKIPPED
        )
    
    return None  # Run detection for other responses


def main():
    parser = argparse.ArgumentParser(description="Run detections on all evaluated experiment files in a folder.")
    parser.add_argument("--evaluated_dir", type=str, required=True, help="Path to the folder containing evaluated experiment documents.")
    parser.add_argument("--detected_dir", type=str, required=True, help="Path to the folder to store detection results.")
    args = parser.parse_args()

    evaluated_dir = Path(args.evaluated_dir)
    detected_dir = Path(args.detected_dir)
    detected_dir.mkdir(parents=True, exist_ok=True)

    # Count total files to process
    evaluated_files = list(evaluated_dir.glob("*.json"))
    total_files = len(evaluated_files)
    print(f"Found {total_files} files to process in {evaluated_dir}")

    for i, evaluated_doc_path in enumerate(evaluated_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {evaluated_doc_path.name}")
        
        detected_filename = evaluated_doc_path.name.replace("evaluated_", "detected_")
        detection_results_path = detected_dir / detected_filename

        rerun_detection = True
        if detection_results_path.exists():
            try:
                detected_doc = DetectedExperimentDocument.load_from_json(detection_results_path)
                has_failed = any(resp.get_failed_detections() for resp in detected_doc.responses)
                if not has_failed:
                    print(f"  Skipping {detection_results_path.name} (no failed detections)")
                    rerun_detection = False
                else:
                    print(f"  Rerunning detection for {detection_results_path.name} (failed detections found)")
            except Exception as e:
                print(f"  Error loading {detection_results_path.name}: {e}. Will rerun detection.")

        
        if rerun_detection:
            print(f"  Running detections for {evaluated_doc_path.name}")
            detected_doc = kon.detect_experiment(
                experiment_document_path=evaluated_doc_path,
                detector_types=[DetectorType.TLM, DetectorType.AWS_BEDROCK, DetectorType.AZURE_CONTENT_SAFETY, DetectorType.DEEPEVAL, DetectorType.RAGAS],
                path_to_store=detection_results_path,
                skip_function=skip_non_abstained_responses
            )
            print(f"  Saved detection results for {evaluated_doc_path.name} to {detection_results_path}")
            print(f"  Total responses detected: {len(detected_doc.responses)}")
            failed_count = 0
            for response in detected_doc.responses:
                for detection in response.detections:
                    if detection.error:
                        print(f"    Detector: {detection.detector_type}")
                        print(f"    Status: {detection.status}")
                        print(f"    Error: {detection.error}")
                    if detection.status == DetectionStatus.FAILED or detection.error:
                        failed_count += 1
            print(f"  Number of failed/error detections: {failed_count} / {len(detected_doc.responses)}")
        
        print(f"  [{i}/{total_files}] Completed: {evaluated_doc_path.name}")
    
    print(f"\nAll {total_files} files processed successfully!")

if __name__ == "__main__":
    main()