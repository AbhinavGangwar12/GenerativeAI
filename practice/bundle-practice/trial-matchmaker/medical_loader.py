import json
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MedicalDataLoader:
    def __init__(self, trials_path: str, guidelines_path: str, patients_path: str):
        self.trials_path = Path(trials_path)
        self.guidelines_path = Path(guidelines_path)
        self.patients_path = Path(patients_path)

    def load_vector_documents(self) -> List[Document]:
        """
        Loads and chunks guidelines.md AND trials.json for the vector store.
        
        Constraints for trials.json:
        1. It is a list of dictionaries.
        2. Format page_content as: "Trial: {title}\nPhase: {phase}\nCriteria: {inclusion_criteria}"
        3. Extract 'trial_id' and 'status' into the metadata. Add {"source": "trial"}.
        """
        if not self.trials_path.is_file():
            raise FileNotFoundError(f"File not found as {self.trials_path}")
        trials = json.loads(self.trials_path.read_text(encoding="utf-8"))
        docs = []
        for trial in trials:
            content = f"Trial: {trial.get('title')}\nPhase: {trial.get('phase')}\nCriteria: {trial.get('inclusion_criteria')}"
            docs.append(Document(page_content=content, metadata = {"trial_id" : trial.get("trial_id") , "status" : trial.get("status"), "source" : "trial"}))
        return docs

    def get_patient_summary(self, patient_id: str) -> str:
        """
        JSON PARSING CHALLENGE:
        Navigate the nested patients.json file to find the specific patient.
        
        Constraints:
        1. The actual patients are inside the "records" list, not at the root.
        2. Extract their age, gender, active medical conditions, and medications.
        3. Format and return it as a single readable string.
        4. Raise a ValueError if the patient_id is not found.
        
        Returns example:
        "Patient is a 45 year old M. Active conditions: Type 2 Diabetes. Medications: Metformin, Albuterol."
        """
        if not self.patients_path.is_file():
            raise FileNotFoundError(f"Patient summary file is absent at {self.patients_path}")
        records = json.loads(self.patients_path.read_text(encoding='utf-8')).get("records")
        # index = any(record.get("patient_id") == patient_id for record in records)
        index = next((i for i, r in enumerate(records) if r.get("patient_id") == patient_id), -1)
        if index == -1:
            raise ValueError("Patient not found!")
        data = records[index]
        medical_conditions = ", ".join([
            cond.get("condition", "None") 
            for cond in data.get("medical_history", []) 
            if cond.get("active")
        ])
        current_medications = ", ".join(data.get("current_medications", []))
        demographics = data.get("demographics", {})
        summary = f"Age: {demographics.get('age')}\nGender: {demographics.get('gender')}\nMedical Conditions: {medical_conditions}\nMedications: {current_medications}"
        return summary