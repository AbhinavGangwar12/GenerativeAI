from typing import List
import json
from pathlib import Path
# Hint: You will likely need to import json and LangChain splitters here
from langchain_core.documents import Document
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

class TechNovaLoader:
    def __init__(self, catalog_path: str, policies_path: str):
        self.catalog_path = Path(catalog_path)
        self.policies_path = Path(policies_path)

    def load_and_chunk(self) -> List[Document]:
        """
        Load and chunk the markdown catalog and JSON policies.

        Constraints:
        1. Use RecursiveCharacterTextSplitter for the markdown file 
           (chunk_size=500, chunk_overlap=50).
        2. Ensure every document chunk has a "source" key in its metadata 
           (value: "catalog" or "policy").
        3. Policy documents must also include a "category" key in their metadata.

        Returns:
            List[Document]: A combined list of all chunked documents.
        """
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")
        if not self.policies_path.is_file():
            raise FileNotFoundError(f"Policies file not found: {self.policies_path}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        catalog_text = self.catalog_path.read_text(encoding="utf-8")
        catalog_chunks = splitter.split_text(catalog_text)
        catalog_docs = [
            Document(page_content=chunk, metadata={"source" : "catalog"})
            for chunk in catalog_chunks
        ]

        policy_docs = []
        with self.policies_path.open("r", encoding="utf-8") as f:
            policies = json.load(f)
            for data in policies:
                category = data.get("category", "unknown")
                title = data.get("title", "unknown")
                content = data.get("content", "")
                policy_id = data.get("id", "unknown")
                text = f"""
                Policy ID: {policy_id}
                Title: {title}
                Category: {category}
                Content: {content}
                """
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    policy_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": "policy", "category": category, "title": title, "id": policy_id}
                        )
                    )
        return catalog_docs + policy_docs