from typing import List
from langchain_core.documents import Document
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter 


class ArchiveLoader:
    def __init__(self, lore_path: str, rules_path: str):
        self.lore_path = Path(lore_path)
        self.rules_path = Path(rules_path)

    def load_and_chunk(self) -> List[Document]:
        """
        Load and chunk the markdown lore and JSON rules.

        Constraints:
        1. Use RecursiveCharacterTextSplitter for the markdown file 
           (chunk_size=400, chunk_overlap=40).
        2. Ensure every document chunk has a "source" key in its metadata 
           (value: "lore" or "rule").
        3. Rule documents must also include a "category" key in their metadata.
        4. Remember to process the JSON list properly so chunks don't inherit 
           the wrong metadata!

        Returns:
            List[Document]: A combined list of all chunked documents.
        """
        # raise NotImplementedError
        if not self.lore_path.is_file():
            raise FileNotFoundError("Lore file not found.")
        if not  self.rules_path.is_file():
            raise FileNotFoundError("Rules file not found.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40, separators=[".", "\n\n", "\n"], keep_separator=False, is_separator_regex=False, len_function=len)

        lore_text = self.lore_path.read_text(encoding="utf-8")
        lore_chunks = splitter.split_text(lore_text)
        lore_docs = [
            Document(page_content=chunk, metadata={"source" : "lore"}) for chunk in lore_chunks
        ]

        rules_docs = []
        with self.rules_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            for rule in data:
                id = rule.get("rule_id", "unknown")
                title = rule.get("title", "unknown")
                category = rule.get("category", "unknown")
                content = f"""
                Rule ID : {id}\n\n
                title: {title} \n\n
                category : {category} \n\n
                content : {rule.get("content", "")}\n\n
                is_active: {rule.get("active", "unknown")}
                """
                chunks = splitter.split_text(content)
                rules_docs.extend(
                    Document(page_content=chunk, metadata={"source": "rule","category" : category, "active" : rule.get("active", "unknown")})
                    for chunk in chunks
                )
        return lore_docs + rules_docs

