import hashlib
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid


qdrand_url = "http://localhost:6333"

class Qdrant:
    def __init__(self, collection_name: str):
        self.client = QdrantClient(url=qdrand_url, timeout=120)
        self.collection_name = collection_name

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        points = [
            PointStruct(
                id=self._make_point_id(
                    chunk["metadata"].get("doc_id", "unknown"), 
                    i
                ),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    **chunk["metadata"]
                }
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        return self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def dense_search(self, query_vector: list[float], top_k: int=5, filter: dict=None) -> list[dict]:
        query_filter = None
        if filter:
            conditions = [
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                )
                for key, value in filter.items()
            ]
            query_filter = Filter(must=conditions)
            
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return [
            {
                "content": r.payload["content"],
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "content"}   
            }
            for r in results.points
        ]
        
    def _make_point_id(self, doc_id: str, chunk_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_index}"))
    
    
    