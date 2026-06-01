from concurrent.futures import ProcessPoolExecutor
from config.ingest_config import get_ingest_config
from tqdm import tqdm   
import hashlib
import json
import os


class Ingest:
    def __init__(self, country: str):
        self.config = get_ingest_config(country)
        
        self.loader = self.config.loader_class()
        self.processor = self.config.processor_class()
        self.embedding_model = self.config.embedding_class()
        
        self.collection_name = self.config.collection_name
        self.vector_db = self.config.vector_db_class(self.collection_name)
        self.vector_db.create_collection()
        
        self.hash_cache_path = self.config.hash_cache_path
        self._load_hash_cache() 
    
    def process(self, batch_size: int=None, offset: int=None):
        if batch_size is None:
            batch_size = self.loader.total_rows()
        if offset is None:
            offset = 0

        df = self.loader.load(batch_size=batch_size, offset=offset)

        rows_to_process = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {self.config.loader_class.__name__} documents"):
            doc_id = str(row[self.config.id_field])
            current_hash = self._content_hash(str(row[self.config.content_field]))
            if self.hash_cache.get(doc_id) == current_hash:
                continue
            doc_metadata = {**self.config.metadata_fields}
            rows_to_process.append((row, doc_metadata, current_hash))

        num_unprocessed = len(rows_to_process)
       

        batch_worker = 10
        if not rows_to_process:
            print("No new documents to process.")
            return

        print(f"Processing {num_unprocessed} documents...")
        batch_size_process = 10000
        total_chunks = 0

        for batch_start in range(0, len(rows_to_process), batch_size_process):
            batch = rows_to_process[batch_start:batch_start+batch_size_process]
            args = [(row, meta, self.config) for row, meta, _ in batch]

            with ProcessPoolExecutor(max_workers=batch_worker) as executor:
                results = list(
                    tqdm(
                        executor.map(_process_row, args),
                        total=len(args),
                        desc=f"Processing batch {batch_start//batch_size_process+1} ({len(batch)} docs)"
                    )
                )

            chunks = []
            for (row, _, current_hash), result in zip(batch, results):
                doc_id = str(row[self.config.id_field])
                chunks.extend(result)
                if doc_id:
                    self.hash_cache[doc_id] = current_hash
                    self._save_hash_cache()  

            if chunks:
                embeddings = self.embedding_model.embed_documents([chunk["content"] for chunk in chunks])
                self.vector_db.upsert_chunks(chunks, embeddings)

            total_chunks += len(chunks)
            print(f"Batch {batch_start//batch_size_process+1}: {len(chunks)} chunks upserted")
        print(f"Done: {total_chunks} chunks upserted")
   
    
    def _load_hash_cache(self):
        if os.path.exists(self.hash_cache_path):
            with open(self.hash_cache_path, "r") as f:
                self.hash_cache = json.load(f)
        else:
            self.hash_cache = {}
            
    def _save_hash_cache(self):
        os.makedirs(os.path.dirname(self.hash_cache_path), exist_ok=True)
        with open(self.hash_cache_path, "w") as f:
            json.dump(self.hash_cache, f)
            
    def _content_hash(self, html: str) -> str:
        return hashlib.md5(html.encode()).hexdigest()

def _process_row(args):
    row, doc_metadata, config = args
    processor = config.processor_class()
    raw_content = row[config.content_field]
    return processor.process(raw_content=raw_content, doc_metadata=doc_metadata)
    

    

        
