from typing import List
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document

class FinanceDocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_and_preprocess(self) -> List[Document]:
        """
        Loads the CSV file and converts it to LangChain Documents.
        
        Core Requirements:
        1. The page_content MUST be formatted exactly as:
           "Title: {title}\nSector: {sector}\nReport: {content}"
        2. The metadata MUST include 'report_id' (str) and 'fiscal_year' (int).
        
        Returns:
            List[Document]: The processed documents.
        """
        # 1. Correct exception for missing file
        if not self.file_path.is_file():
            raise FileNotFoundError(f"File not found at {self.file_path}")
        
        # 2. Load the CSV data
        df = pd.read_csv(self.file_path)
        return_docs = []
        
        # 3. Use itertuples() for faster execution and cleaner row access
        for row in df.itertuples(index=False):
            # Format the page content exactly to your specifications
            page_content = f"Title: {row.title}\nSector: {row.sector}\nReport: {row.content}"
            
            # Enforce data types for metadata as requested (str and int)
            metadata = {
                "report_id": str(row.report_id),
                "fiscal_year": int(row.fiscal_year)
            }
            
            # 4. Use .append() to add individual Document objects to the list
            return_docs.append(Document(page_content=page_content, metadata=metadata))
            
        return return_docs
