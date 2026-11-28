"""Inventory candidate binary fields from pinned Mega-Study questionnaire metadata."""
from hashlib import sha256
import json
from urllib.error import URLError
from urllib.request import urlopen

REVISION = "0401b715a341ac4b5f98b4424b4aecf9d29570d0"
DATASET = "LLM-Digital-Twin/Twin-2K-500-Mega-Study"
STUDIES = (
    "accuracy_nudges", "affective_priming", "consumer_minimalism", "context_effects",
    "default_eric", "digital_certification", "hiring_algorithms", "idea_evaluation",
    "idea_generation", "infotainment", "junk_fees", "obedient_twins",
    "preference_redistribution", "privacy", "promiscuous_donors",
    "quantitative_intuition", "recommendation_algorithms", "story_beliefs",
    "targeting_fairness",
)


def inventory(raw):
    entries = json.loads(raw)["SurveyElements"]
    candidates = []
    for entry in entries:
        question = entry["Payload"]
        if (entry["Element"] == "SQ" and question.get("QuestionType") == "MC"
                and len(question.get("Choices", {})) == 2):
            candidates.append(entry["PrimaryAttribute"])
    return {"questionnaire_sha256": sha256(raw).hexdigest(),
            "binary_choice_candidates": sorted(candidates)}


def main():
    result = {"revision": REVISION, "studies": {}, "unavailable": []}
    for study in STUDIES:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/.dat/{study}/raw_data/survey.qsf"
        try:
            with urlopen(url, timeout=12) as source:
                result["studies"][study] = inventory(source.read())
        except (URLError, TimeoutError):
            result["unavailable"].append(study)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
