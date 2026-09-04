from typing import Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json
from pathlib import Path
from .archive_store import ArchiveVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ArchivistAgent:
    def __init__(self, vector_store: ArchiveVectorStore, stats_db_path: str):
        """
        Initialize the vector store, load the knight stats JSON database, 
        and set up a mechanism to track chat history.
        """
        self.vector_store = vector_store
        self.chat_history: list[BaseMessage] = []
        # TODO: Load knight stats database from stats_db_path
        with Path(stats_db_path).open("r", encoding="utf-8") as f:
            self.stats_db = json.load(f)    
        # raise NotImplementedError
        self.llm = ChatOllama(model="llama3")

    def _contextualize_query(self, user_query: str) -> str:
        """
        Rewrite the user's query to be a standalone question based on chat history.

        Args:
            user_query: The latest input from the user.

        Returns:
            str: A standalone question that makes sense without history.
        """
        if not self.chat_history:
            return user_query
        history_text = "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in self.chat_history
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just rewrite it if needed, otherwise return it as is."),
            ("human", "Chat history:\n{history}\n\nUser_query:\n{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history" : history_text, "query" : user_query})

    def _route_query(self, standalone_query: str) -> str:
        """
        Classify the query into a specific routing category using an LLM.

        Args:
            standalone_query: The contextualized question.

        Returns:
            str: Strictly either "KNIGHT_STATS" or "GENERAL_LORE".
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a classifier. Classify the user query into strictly one of two categories: 'KNIGHT_STATS' or 'GENERAL_LORE'. Respond with ONLY the category name."),
            ("human", "{query}")
        ])
        p = prompt.format({"query" : standalone_query})
        chain = prompt | self.llm | StrOutputParser()
        verdict = chain.invoke({"query": standalone_query})
        return "KNIGHT_STATS" if "KNIGHT_STATS" in verdict else "GENERAL_LORE"

    async def chat(self, user_query: str, knight_id: Optional[str] = None) -> str:
        """
        Process the user query through the full RAG pipeline.

        Steps:
        1. Contextualize the question using history.
        2. Route the standalone question.
        3. If "KNIGHT_STATS": Bypass vector search. Fetch data from the local 
           stats dict using the knight_id, construct a prompt, and generate an answer.
        4. If "GENERAL_LORE": Perform an advanced_search on the vector store, 
           construct a prompt with the context, and generate an answer.
        5. Update self.chat_history with both the query and the final response.

        Args:
            user_query: The user's message.
            knight_id: Optional ID for database lookups (e.g., "KNT-001").

        Returns:
            str: The AI-generated response.
        """
        # raise NotImplementedError
        _contextualize_query = self._contextualize_query(user_query=user_query)
        decision = self._route_query(standalone_query=_contextualize_query)

        if decision == "KNIGHT_STATS":
            if knight_id and knight_id in self.stats_db:
                knight_info = self.stats_db[knight_id]
                template = ChatPromptTemplate.from_template(
                    "You are a support agent. Answer the query using the Knight details.\n\n"
                    "Order Info:\n{knight_info}\n\n"
                    "Query: {query}"
                )
                chain = template | self.llm | StrOutputParser()
                response = chain.invoke({"knight_info": json.dumps(knight_info, indent=2), "query": _contextualize_query})
            else:
                raise ValueError("Knight Id not found.")
        else:
            docs = await self.vector_store.advanced_search(_contextualize_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template(
                "You are the maeter of Citadel. Answer the question accurately using the context below.\n\n"
                "<context>:\n{context}</context>\n\n"
                "Question: {query}\n\n"
                "Answer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context" : context, "query" : _contextualize_query})
        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content=response))
        return response

