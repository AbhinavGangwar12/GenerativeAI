import json
from pathlib import Path
from .config import Config
from .cyber_store import ThreatVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class InvestigatorAgent:
    # PROMPTS PROVIDED FOR YOU
    HYDE_PROMPT = """You are a cybersecurity expert. Write a detailed, hypothetical threat intelligence report based on the following brief security alert. 
    Assume the worst-case scenario. Use technical terminology. Do not include introductory filler.
    
    Alert Description: {alert_description}
    
    Hypothetical Report:"""

    FINAL_ANALYSIS_PROMPT = """You are a SOC Analyst. Based on the actual threat intelligence context provided, analyze the security alert and identify the likely threat actor.
    
    Real Threat Intel Context:
    {context}
    
    Security Alert: {alert_description}
    
    Final Analysis:"""

    def __init__(self, vector_store: ThreatVectorStore, alerts_path: str):
        self.vector_store = vector_store
        
        # TODO: Load incident_logs.json into self.alerts_db
        with Path(alerts_path).open("r") as f:
            self.alerts_db = json.load(f)
        # TODO: Initialize self.llm using Config.get_api_key()
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=Config.get_api_key())

    async def _generate_hypothetical_doc(self, alert_description: str) -> str:
        """
        CURVEBALL Step 1: Generate the Fake Document (HyDE).
        Pass the alert_description into the HYDE_PROMPT and return the LLM's output.
        """
        template = ChatPromptTemplate.from_template(self.HYDE_PROMPT)
        chain = template | self.llm | StrOutputParser()
        return await chain.ainvoke({"alert_description" : alert_description})

    async def investigate_alert(self, alert_id: str) -> str:
        """
        The Main Pipeline:
        1. Look up the alert_id in self.alerts_db. (Return error string if not found).
        2. Extract the "description" from the alert.
        3. Pass the description to `_generate_hypothetical_doc()` to get a fake, detailed report.
        4. CRITICAL: Pass the FAKE report (not the original short description) 
           into `self.vector_store.search_intel()` to fetch the real documents.
        5. Combine the retrieved real documents into a single context string.
        6. Pass the context and the ORIGINAL alert description into the 
           FINAL_ANALYSIS_PROMPT to get the ultimate answer.
        
        Returns:
            str: The final SOC analysis.
        """
        index = next((i for i,r in enumerate(self.alerts_db) if r.get("alert_id") == alert_id), -1)
        if index == -1:
            raise ValueError("Alert Id not found.")
        data = self.alerts_db[index]
        desc = data.get("description", '')
        detailed_rep = await self._generate_hypothetical_doc(alert_description=desc)
        real_docs = await self.vector_store.search_intel(query=detailed_rep)
        context_string = "\n".join([doc.page_content for doc in real_docs])
        template = ChatPromptTemplate.from_template(self.FINAL_ANALYSIS_PROMPT)
        chain = template | self.llm | StrOutputParser()
        return await chain.ainvoke({"context" : context_string, "alert_description" : desc})