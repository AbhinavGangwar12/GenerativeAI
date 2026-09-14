import json
import uuid
from typing import List, Dict, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid 

class EcommerceLoader:
    def __init__(self, products_path: str, policies_path: str):
        self.products_path = products_path
        self.policies_path = policies_path

    def get_id(self):
        return uuid.uuid4().hex
    def process_data(self) -> Tuple[Dict[str, str], List[Document]]:
        """
        PARENT-CHILD CHALLENGE:
        1. Load `policies.md` and chunk it normally (assign unique parent_ids to these chunks too).
        2. Load `products.json`. For each product, convert the entire complex JSON 
           object into a single readable string (The Parent).
        3. Generate a unique `parent_id` (using uuid) for each product.
        4. Save the full product string into a Python Dictionary with the `parent_id` as the key.
        5. Chunk the product string into smaller Documents (The Children).
        6. Inject the `parent_id` into the metadata of every Child chunk.
        
        Returns:
            Tuple containing:
            1. parent_db: Dict[str, str] mapping parent_ids to full document strings.
            2. child_docs: List[Document] containing all the small chunks.
        """
        if not Path(self.policies_path).is_file():
            raise ValueError("Policies not found!")
        if not Path(self.products_path).is_file():
            raise ValueError("Products not found!")

        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=30)
        parent_db = {}
        docs = []
        
        policies = Path(self.policies_path).read_text(encoding="utf-8")
        policy_chunks = splitter.split_text(policies)
        for chunk in policy_chunks:
            p_id = self.get_id()
            parent_db[p_id] = chunk  # Save to parent_db!
            docs.append(Document(page_content=chunk, metadata={"parent_id": p_id}))

        with Path(self.products_path).open("r") as f:
            products = json.load(f)

        
        for product in products:
            p_id = self.get_id() # Fix: Added parentheses
            parent_db[p_id] = json.dumps(product, indent=4)

        for key, value in parent_db.items():
            # Skip re-chunking policies (we already chunked them above)
            if "{" in value: # A hacky but effective way to know it's the JSON product
                product_chunks = splitter.split_text(value)
                docs.extend([
                    Document(page_content=c, metadata={"parent_id": key}) for c in product_chunks 
                ])

        return parent_db, docs

        