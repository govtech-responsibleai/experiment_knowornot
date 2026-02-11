import os
from pathlib import Path
from knowornot.Detector.models import DetectedExperimentDocument

def main():
    detected_dir = Path("experiments/detected")
    target_types = ["deepeval", "gemini"]
    files = list(detected_dir.glob("*.json"))
    print(f"Found {len(files)} files in {detected_dir}")
    for file in files:

        doc = DetectedExperimentDocument.load_from_json(file)
        
        responses_new = []
        for response in doc.responses:
            # Remove from detections
            new_response_dict = response.model_copy()
            new_response_dict.detections = [
                d for d in response.detections
                if getattr(d, "detector_type", "").lower() not in target_types
            ]
            responses_new.append(new_response_dict)
        doc.responses = responses_new
 
        # Remove from detection_metadata
        meta = doc.detection_metadata
        for k in target_types:
            del meta[k]
            
        doc.detection_metadata = meta


        Path(file).write_text(doc.model_dump_json(indent=2))
        # doc.save_to_json(file)
        print(f"Updated {file}")


if __name__ == "__main__":
    main() 