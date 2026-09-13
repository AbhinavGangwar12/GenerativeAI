from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
class ThreatLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_and_chunk(self) -> List[Document]:
        """
        CORE REQUIREMENT:
        1. Load the markdown file.
        2. Chunk using RecursiveCharacterTextSplitter (chunk_size=300, chunk_overlap=30).
        3. Add metadata {"source": "threat_intel"}.
        """
        if not Path(self.filepath).is_file():
            raise FileNotFoundError("File not found!")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        text = Path(self.filepath).read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata = {"source" : "threat_intel"}) for chunk in chunks 
        ]
        return docs
        