from typing import List
from .medical_store import MedicalVectorStore
from .medical_loader import MedicalDataLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MatchmakerAgent:
    def __init__(self, vector_store: MedicalVectorStore, data_loader: MedicalDataLoader):
        self.vector_store = vector_store
        self.data_loader = data_loader
        self.llm = ChatOllama(model="llama3.1") # Or whichever model you are using locally

    async def _generate_search_queries(self, patient_summary: str) -> List[str]:
        """
        QUERY EXPANSION CHALLENGE:
        Pass the patient_summary to the LLM. Ask it to generate exactly 2 distinct 
        search queries to find clinical trials that match the patient's conditions.
        
        Constraints:
        1. Force the LLM to output ONLY the queries, separated by newlines.
        2. Parse the output into a valid Python List[str].
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpul medical assistant, you are provided with some patient summary and you have to provide exactly 2 distinct search queries to find clinical trials that match the patient's conditions.\n\nRULES: SEPARATE THE TWO QUERIES WITH NEWLINE CHARACTER i.e. \\n"),
            ("human", "summary:\n\n{summary}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"summary" : patient_summary})
        return response.split("\n")

    async def find_trials(self, patient_id: str) -> str:
        """
        The Main Pipeline:
        1. Call self.data_loader.get_patient_summary(patient_id).
        2. Generate search queries based on that summary.
        3. Iterate through the queries, fetch documents from the vector store, 
           and deduplicate them using a Python Set (use page_content as the uniqueness key).
        4. Pass the patient summary AND the deduplicated trial documents to the LLM 
           to recommend the best trial.
        
        Returns:
            str: The final recommendation from the LLM.
        """
        patient_summary = self.data_loader.get_patient_summary(patient_id=patient_id)
        search_queries = await self._generate_search_queries(patient_summary=patient_summary)
        lookup = set()
        docs = []
        for query in search_queries:
            top_k = await self.vector_store.search(query=query)
            for doc in top_k:
                if doc.page_content not in lookup:
                    lookup.add(doc.page_content)
                    docs.append(doc)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant, you are provided with the patient summary and the patient's trial history, based on these things, generate a short but best recommendation for the next trial."),
            ("human", "<SUMMARY>\n{summary}\n</SUMMARY>\n\nTrial-history:\n{trial}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        trial_context = "\n\n".join([doc.page_content for doc in docs])
        response = await chain.ainvoke({"summary" : patient_summary, "trial" : trial_context})
        return response
            