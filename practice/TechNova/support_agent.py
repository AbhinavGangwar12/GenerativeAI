from typing import Dict, List, Optional
from .hybrid_store import TechNovaVectorStore
import json
from pathlib import Path
# Hint: You will likely need ChatPromptTemplate, ChatOpenAI, and an output parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from .hybrid_store import TechNovaVectorStore

from pathlib import Path
import json

class ConversationalSupportAgent:
    def __init__(self, vector_store: TechNovaVectorStore, order_db_path: str):
        """
        Initialize the vector store, load the order JSON database, 
        and set up a mechanism to track chat history.
        """
        self.vector_store = vector_store
        self.chat_history = []
        # TODO: Load order database from order_db_path
        with Path(order_db_path).open("r", encoding="utf-8") as f:
            self.order_db = json.load(f)

        self.llm = ChatOllama(model="llama3")
        # raise NotImplementedError

    def _condense_question(self, user_query: str) -> str:
        """
        Rewrite the user's query to be a standalone question based on chat history.

        Args:
            user_query: The latest input from the user.

        Returns:
            str: A standalone question that makes sense without history.
        """
        if not self.chat_history:
            return user_query
        history_str = "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'} : {msg.content}" for msg in self.chat_history
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just rewrite it if needed, otherwise return it as is."),
            ("human", "Chat History:\n{history}\n\nQuestion: {query}\n\nStandalone Question:")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"history" : history_str, "query" : user_query})
    
    def _route_query(self, standalone_query: str) -> str:
        """
        Classify the query into a specific routing category using an LLM.

        Args:
            standalone_query: The contextualized question.

        Returns:
            str: Strictly either "ORDER_STATUS" or "GENERAL_SUPPORT".
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a classifier. Classify the user query into strictly one of two categories: 'ORDER_STATUS' or 'GENERAL_SUPPORT'. Respond with ONLY the category name."),
            ("human", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        decision = chain.invoke({"query" : standalone_query})
        return "ORDER_STATUS" if "ORDER_STATUS" in decision else "GENERAL_SUPPORT"

    async def chat(self, user_query: str, order_id: str = None) -> str:
        """
        Process the user query through the full RAG pipeline.

        Steps:
        1. Condense the question using history.
        2. Route the condensed question.
        3. If "ORDER_STATUS": Bypass vector search. Fetch data from the local 
           order dict using the order_id, construct a prompt, and generate an answer.
        4. If "GENERAL_SUPPORT": Perform an advanced_search on the vector store, 
           construct a prompt with the context, and generate an answer.
        5. Update self.chat_history with both the query and the final response.

        Args:
            user_query: The user's message.
            order_id: Optional ID for order lookups.

        Returns:
            str: The AI-generated response.
        """
        standalone_query = self._condense_question(user_query)
        route = self._route_query(standalone_query)

        if route == "ORDER_STATUS":
            if order_id and order_id in self.order_db:
                order_info = self.order_db[order_id]
                prompt = ChatPromptTemplate.from_template(
                    "You are a customer support agent. Answer the customer's query using their order details.\n\n"
                    "Order Info:\n{order_info}\n\n"
                    "Query: {query}"
                )
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({"order_info": json.dumps(order_info, indent=4), "query": standalone_query})
            else:
                response = "Order ID not found or not provided. Please provide a valid order ID for order status inquiries."
        else:
            docs = await self.vector_store.advanced_search(standalone_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template(
                "You are TechNova's expert customer support agent. Answer the question accurately using the context below.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Answer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context, "query": standalone_query})

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content=response))
        return response

