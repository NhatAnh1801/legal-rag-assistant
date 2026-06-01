from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from tqdm import tqdm
import numpy as np
import statistics
import time
import uuid

qdrand_url = "http://localhost:6333"

class Qdrant:
    def __init__(self, collection_name: str):
        self.client = QdrantClient(url=qdrand_url, timeout=300)
        self.collection_name = collection_name
        self.vector_size = 128  # Temporary set to Vietnamese embedding dimension

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]], batch_size: int = 6200):
        total = len(chunks)
        results = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = []
            for chunk, embedding in tqdm(
                zip(chunks[start:end], embeddings[start:end]), 
                total=end - start, 
                desc=f"Creating Qdrant points ({start}-{end})"
            ):
                point = PointStruct(
                    id=self._make_point_id(
                        chunk["metadata"].get("id"), 
                        chunk["metadata"].get("node_id")
                    ),
                    vector=embedding,
                    payload={
                        "content": chunk["content"],  
                        **chunk["metadata"]
                    }
                )
                points.append(point)
            
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            results.append(result)
        return results
   
    
    def dense_search(self, query_vector: list[float], top_k: int=None, filter: dict=None) -> list[dict]:
        if top_k is None:
            top_k = self.top_k
            
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
        
    def _make_point_id(self, id: str, node_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{id}_{node_id}"))
    
    def _find_optimal_batch_size(self, candidates: list[int], total_points: int = 10000, repeat: int = 5) -> dict:
        """
        Find the optimal batch size for upserting points to Qdrant by empirically testing
        Args:
            candidates (list[int]): A list of batch sizes to evaluate.
            total_points (int): The total number of points to upsert in each trial (default: 5000).
            repeat (int): The number of upsert repetitions to perform for each batch size (default: 5).

        Returns:
            dict: A dictionary with the optimal batch size for mean and median upsert time,
                as {"mean": optimal_batch_size, "median": optimal_batch_size}.
        """
        max_words = 500 
        
        fake_word = "từ"
        fake_sentence = " ".join([fake_word] * 12) + "."
        fake_sent_words = fake_sentence.split()
        n_full_sent = max_words // len(fake_sent_words)
        remain = max_words % len(fake_sent_words)
        content_words = []
        content_words.extend(fake_sent_words * n_full_sent)
        if remain:
            content_words.extend(fake_sent_words[:remain])
        content = " ".join(content_words)
        vec = np.full((128,), 1.0, dtype=np.float16)
   
        sample_points = []
        for i in range(total_points):
            point = {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "content": content,
                    "metadata": {
                        "node_id": f"node_id_{i}",
                        "header_name": f"header_name_{i}", 
                        "header_value": f"header_value_{i}",
                        "header_index": i,
                        "level": f"level_{i}",
                        "title": f"title_{i}",
                        "so_ky_hieu": f"so_ky_hieu_{i}",
                        "loai_van_ban": f"loai_van_ban_{i}",
                        "ngay_ban_hanh": "2024-01-01",
                        "co_quan_ban_hanh": f"co_quan_ban_hanh_{i}",
                        "linh_vuc": f"linh_vuc_{i}",
                        "nganh": f"nganh_{i}",
                        "ngay_co_hieu_luc": "2024-01-02",
                        "ngay_het_hieu_luc": "2025-01-02",
                        "nguoi_ky": f"nguoi_ky_{i}",
                        "pham_vi": f"pham_vi_{i}",
                    }
                }
            }
            sample_points.append(point)

        results = {}  
        for batch_size in candidates:
            batch_times = []
            for run in range(repeat):
                batches = [
                    sample_points[i:i + batch_size]
                    for i in range(0, total_points, batch_size)
                ]
                start = time.time()
                for batch in batches:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                duration = time.time() - start
                batch_times.append(duration)
                print(f"[Batch size {batch_size}] Run {run+1}/{repeat} - Upsert time: {duration:.4f}s")
            results[batch_size] = batch_times
            means = statistics.mean(batch_times)
            medians = statistics.median(batch_times)
            print(f"Batch size: {batch_size}, Mean upsert time: {means:.4f}s, Median upsert time: {medians:.4f}s over {repeat} runs")

        best_mean = min(candidates, key=lambda b: statistics.mean(results[b]))
        best_median = min(candidates, key=lambda b: statistics.median(results[b]))
        print(f"Optimal batch size for mean: {best_mean}")
        print(f"Optimal batch size for median: {best_median}")
        return {"mean": best_mean, "median": best_median}


        