from typing import List
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class LegalDocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def generate_uuid(self):
        return uuid.uuid4().hex

    def load_and_preprocess(self) -> List[Document]:
        """
        Loads the markdown contract and chunks it.
        
        Core Requirements:
        1. Use RecursiveCharacterTextSplitter (chunk_size=300, chunk_overlap=30).
        2. CRITICAL FOR DEDUPLICATION: You must inject a unique 'chunk_id' into 
           the metadata of EVERY document chunk (e.g., using uuid.uuid4().hex).
        
        Returns:
            List[Document]: The processed documents with unique IDs.
        """
        if not Path(self.file_path).is_file():
            raise FileNotFoundError(f"File not found at ${self.file_path}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators=[".", "\n\n", "\n"], len_function=len, is_separator_regex=False)
        doc = Path(self.file_path).read_text(encoding="utf-8")
        chunks = splitter.split_text(doc)
        chunked_docs = [
            Document(page_content=chunk, metadata={"chunk_id" : self.generate_uuid()}) for chunk in chunks 
        ]
        return chunked_docs