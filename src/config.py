import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROUND_TRUTH_PATH = os.path.join("data", "ground_truth.csv")
FHIR_RECORDS_DIR = os.path.join("data", "fhir_records")
FHIR_STORE_PATH = os.path.join("data", "fhir_store.db")
EVAL_RESULTS_DIR = os.path.join("data", "eval_results")

FHIR_RESOURCE_TYPES = [
    "Patient",
    "Condition",
    "Observation",
    "MedicationRequest",
    "Procedure",
    "Encounter",
    "DiagnosticReport",
    "MedicationAdministration",
]
