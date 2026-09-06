import json
from pathlib import Path
from typing import Dict
from .sre_store import RunbookVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

class SRECopilot:
    def __init__(self, vector_store: RunbookVectorStore, alerts_db_path: str):
        self.vector_store = vector_store
        
        # Load active alerts database
        with Path(alerts_db_path).open("r", encoding="utf-8") as f:
            self.alerts_db: Dict = json.load(f)
            
        # TODO: Initialize your LLM here
        self.llm = ChatOllama(model="llama3.1")

    async def diagnose_alert(self, alert_id: str) -> str:
        """
        Perform a Chained Retrieval to diagnose an active alert.

        Steps:
        1. Look up the alert_id in self.alerts_db. If not found, return an error string.
        2. Extract the "issue" (e.g., "CrashLoopBackOff") from the alert dictionary.
        3. Use the extracted "issue" to perform an async search against the vector_store.
        4. Combine the raw alert dictionary data AND the retrieved runbook context into a prompt.
        5. Invoke the LLM to generate a customized, step-by-step terminal command guide 
           specifically tailored to the failing pod/node.

        Args:
            alert_id: The ID of the alert (e.g., "ALT-404").

        Returns:
            str: The AI-generated troubleshooting steps.
        """
        if not alert_id or alert_id not in self.alerts_db:
            return f"Error: Alert {alert_id} not found in active cluster database."
            
        alert_data = self.alerts_db[alert_id]
        issue = alert_data.get("issue", "")
        
        docs = await self.vector_store.search_runbooks(query=issue)
        
        context_text = "\n\n".join(
            [f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs]
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an SRE copilot expert. Your task is to provide a step-by-step terminal command guide specifically tailored to the failing pod/node."),
            ("human", "Alert ID: {alert_id}\n\nAlert Info:\n{info}\n\nRunbook Context:\n{context}")
        ])
        
        chain = prompt_template | self.llm | StrOutputParser()
        
        return chain.invoke({
            "alert_id": alert_id, 
            "info": json.dumps(alert_data, indent=2), 
            "context": context_text
        })
