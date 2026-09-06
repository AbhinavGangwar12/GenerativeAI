from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathLib import Path

class RunbookLoader:
    def __init__(self, runbook_path: str):
        self.runbook_path = Path(runbook_path)

    def load_and_chunk(self) -> List[Document]:
        """
        Load and chunk the markdown runbooks.

        Constraints:
        1. Use an appropriate text splitter (e.g., MarkdownHeaderTextSplitter or 
           RecursiveCharacterTextSplitter) to chunk the runbooks.
        2. Ensure the chunk metadata contains {"source": "runbook"}.

        Returns:
            List[Document]: Chunked runbook documents.
        """
        if not self.runbook_path.is_file():
            raise FileNotFoundError("File not found")
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, len_function=len, is_separator_regex=False)
        docs = self.runbook_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(docs)
        runbook_docs = [
            Document(page_content=chunk, metadata={"source" : "runbook"}) for chunk in chunks
        ]
        return runbook_docs