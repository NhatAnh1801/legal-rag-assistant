import re
from typing import Counter
from src.models.embeddings.gte_multi_base import GTE    # Temporary

from src.processing.vn import VietnameseDocumentProcessor
from src.data_loader.vn import VietnameseDataLoader
from src.qdrant import Qdrant

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm   
import hashlib
import json
import os
import time

LOADER = {"vn": VietnameseDataLoader}
PROCESSOR = {"vn": VietnameseDocumentProcessor}
VECTOR_DB = {"vn": ("vn_documents", Qdrant)}
HASH_CACHE_PATH = {"vn": "./data/cache/vn_hashes.json"}

class Ingest:
    def __init__(self, country: str, embedding_model):
        self.loader = LOADER[country]()
        self.processor = PROCESSOR[country]()
        self.embedding_model = embedding_model
        self.country = country
        
        # Vector database
        collection_name, qdrant_cls = VECTOR_DB[country]
        self.vector_db = qdrant_cls(collection_name)
        self.vector_db.create_collection()
    
        # Cache
        self.hash_cache_path = HASH_CACHE_PATH[country]
        self._load_hash_cache()
    
    def process(self, batch_size: int=None, offset: int=None):
        df = self.loader.load(batch_size=batch_size, offset=offset)
        
        rows_to_process = []
        # Check if the content has been processed
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {self.country} documents"):
            doc_id = str(row["id"])
            current_hash = self._content_hash(row["content_html"])
            if self.hash_cache.get(doc_id) == current_hash:
                continue
            doc_metadata = {
                "doc_id": str(row["id"]),
                "title": row["title"],
                "loai_van_ban": row["loai_van_ban"],
                "country": self.country
            }
            rows_to_process.append((row, doc_metadata, current_hash))
        
        # Process the rows needed to be processed
        args = [(row, meta) for row, meta, _ in rows_to_process]
        # start = time.time()
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_process_row, args), total=len(args), desc=f"Processing {self.country} documents"))
        
        chunks = []
        for idx, (result, (_, _, current_hash)) in enumerate(zip(results, rows_to_process)):
            chunks.extend(result)
            doc_id = result[0]["metadata"]["doc_id"] if result else None
            if doc_id:
                self.hash_cache[doc_id] = current_hash
        
        # Embed the chunks and upsert to vector database if there are chunks to process
        if chunks:
            embeddings = self.embedding_model.embed_documents([chunk["content"] for chunk in chunks])
            self.vector_db.upsert_chunks(chunks, embeddings)
            
        self._save_hash_cache()
        print(f"Done: {len(chunks)} chunks upserted")
    
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
    row, doc_metadata = args
    processor = VietnameseDocumentProcessor()
    return processor.process(raw_content=row["content_html"], doc_metadata=doc_metadata)

def _count_duplicates(client, collection_name: str, limit: int = 500):
    all_points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=True
    )
    
    def norm_text(x: str) -> str:
        return re.sub(r"\s+", " ", (x or "")).strip().lower()
    
    pairs = []
    for p in tqdm(all_points, desc="Processing points"):
        doc_id = p.payload.get("doc_id")
        content = p.payload.get("content")
        if doc_id is None or content is None:
            continue
        pairs.append((str(doc_id), norm_text(content)))
        
    pair_counts = Counter(pairs)
    dup_pairs = {k: v for k, v in pair_counts.items() if v > 1}
    print("duplicate (doc_id, content) groups:", len(dup_pairs))
    print("duplicate rows in those groups:", sum(dup_pairs.values()))
    for i, ((doc_id, content_norm), cnt) in enumerate(dup_pairs.items(), 1):
        print("-" * 100)
        print(f"{i}. doc_id={doc_id} appears {cnt} times")
        print(f"content_preview={content_norm[:300]}")
        if i >= 20:
            break

#if __name__ == "__main__":
    #embedding_model = GTE()
    
    # ingest = Ingest(country="vn", embedding_model=embedding_model)
    # BATCH = 5000
    # total = ingest.loader.total_rows()
    # for offset in range(0, total, BATCH):
    #     ingest.process(batch_size=BATCH, offset=offset)
    
    # ingest = Ingest(country="vn", embedding_model=embedding_model)
    # vector_db = Qdrant("vn_documents")
    #query_vector = embedding_model.embed_query("người lao động phải đóng bảo hiểm xã hội bao nhiêu tiền?")
    # start = time.time()
    # results = vector_db.dense_search(query_vector, top_k=5)
    # elapsed = time.time() - start
    # print(f"elapsed: {elapsed:.1f}s")
    # for r in results:
    #     print("-"*100)
    #     print(r["metadata"]["title"])
    #     print(r["content"])
    # collection_info = ingest.vector_db.client.get_collection(ingest.vector_db.collection_name)
    # print(f"collection size: {collection_info.points_count}")


    
    

    

        
