from .config import Config
from .ecommerce_store import ParentChildVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict

class ECommerceSupportAgent:
    def __init__(self, vector_store: ParentChildVectorStore, parent_db: Dict[str, str]):
        self.vector_store = vector_store
        self.parent_db = parent_db
        # TODO: Initialize LLM with API Key from Config
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=Config.get_api_key())

    async def answer_question(self, user_query: str) -> str:
        """
        The Main Pipeline:
        1. Call retrieve_parents() on the vector store, passing the user_query and self.parent_db.
        2. Combine the returned FULL parent documents into a context string.
        3. Prompt the LLM to answer the user's question accurately based ONLY on the context.
        
        Returns:
            str: The AI's response.
        """
        parents = await self.vector_store.retrieve_parents(query=user_query, parent_db=self.parent_db)
        context_docs = "\n\n".join(context for context in parents)
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful support agent. Answer the query using ONLY the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Query: {query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({"context": context_docs, "query": user_query})