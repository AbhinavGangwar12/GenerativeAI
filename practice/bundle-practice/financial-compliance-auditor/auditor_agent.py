import json
from pathlib import Path
from .config import Config
from .compliance_store import ComplianceVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Hint: You will need ChatPromptTemplate, ChatOpenAI, StrOutputParser

class AuditorAgent:
    def __init__(self, vector_store: ComplianceVectorStore, trades_path: str):
        self.vector_store = vector_store
        
        # Load trades JSON
        with Path(trades_path).open("r", encoding="utf-8") as f:
            self.trades_db = json.load(f)
            
        # TODO: Load your API key using Config and initialize the LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=Config.get_llm_api_key())
        # (Pass the api_key explicitly to the LLM constructor)
        

    async def _grade_document(self, trade_summary: str, policy_context: str) -> str:
        """
        SELF-REFLECTION CHALLENGE (Step 1):
        Ask the LLM to act as a grader. It must determine if the `policy_context` 
        contains the necessary rules to evaluate the `trade_summary`.
        
        Constraints:
        1. Force the LLM to output strictly "YES" or "NO". Do not let it output 
           anything else.
           
        Returns:
            str: "YES" or "NO"
        """
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a strict policy auditor. Your task is to analyze the given trade summary and policy context which is provided. Now your task is to determine if the 'policy_context contains the necessary rules to evaluate the 'trade_summary' or not.\n\nRULES:\n1. You are allowed to give one word answer.\t'YES' if the 'policy_context' contains the necessary rules to evaluate the 'trade_summary', otherwise 'NO'.\n2. Any other type of answer will not be accepted."),
            ("human", "POLICY_CONTEXT:\n{policy}\n\nTRADE_SUMMARY:\n{trade}")
        ])
        chain = template | self.llm | StrOutputParser()
        return await chain.ainvoke({"policy" : policy_context , "trade": trade_summary})

    async def audit_trade(self, trade_id: str) -> str:
        """
        The Main Pipeline:
        1. Look up the trade in self.trades_db (Return error if not found).
        2. Formulate a search query based on the trade's 'asset'.
        3. Fetch the policy context from the vector store.
        4. Pass the trade details and the policy to `_grade_document`.
        5. If the grader returns "NO", immediately return the exact string: 
           "ESCALATE TO COMPLIANCE OFFICER".
        6. If the grader returns "YES", invoke a second LLM chain to generate 
           the final compliance audit report.
           
        Returns:
            str: The audit report OR the escalation string.
        """
        index = next((i for i, r in enumerate(self.trades_db) if r.get("trade_id") == trade_id), -1)
        if index == -1:
            raise ValueError(f"trade_id {trade_id} not found in the db.")
        data = self.trades_db[index]
        query = data.get("asset", "")
        policy_context = await self.vector_store.get_policy(query=query)
        trade_summary = json.dumps(data, indent=2)
        verdict = await self._grade_document(trade_summary=trade_summary, policy_context=policy_context)
        if verdict.lower() == "no":
            return "ESCALATE TO COMPLIANCE OFFICER"
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Finance Auditor, provide the information to the client based on the trade_summary and policy_context (from our trade rules)."),
            ("human", "trade_summary:{trade}\n\npolicy_context:{policy}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({"trade" : query, "policy" : policy_context})
