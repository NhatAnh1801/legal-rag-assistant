from src.data_loader.vn import VietnameseDataLoader
from src.processing.vn import VietnameseDocumentProcessor
from src.models.embeddings.vn_law_embedding import VNLawEmbedding
from src.models.embeddings.vietnamese_embedding import VietnameseEmbedding
from src.models.embeddings.gte_multi_base import GTE

import numpy as np
from tqdm import tqdm
import json
import os
import time
import pandas as pd

EMBEDDING_MODELS = {
    "VietnameseEmbedding": VietnameseEmbedding(),
    "VNLawEmbedding": VNLawEmbedding(),
    "GTE": GTE()
}

def evaluate_embedding_models():
    chunks_path = os.path.join(os.path.dirname(__file__), "dataset", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    queries_path = os.path.join(os.path.dirname(__file__),"dataset","query.json")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    # Chunk lookup
    docid_to_chunks = {}
    docid_to_chunks_list = {}
    for doc in chunks_data:
        cdict = {}
        clist = []
        for chunk in doc["chunks"]:
            cdict[chunk["id"]] = chunk["content"]
            clist.append(chunk)
        docid_to_chunks[doc["id"]] = cdict
        docid_to_chunks_list[doc["id"]] = clist

    results = []

    TOP_K_VALUES = [1, 3, 5, 10]

    for model_name, model in EMBEDDING_MODELS.items():
        hit_counts = {k: 0 for k in TOP_K_VALUES}
        reciprocal_ranks = []
        ndcgs = []
        start_time = time.time()

        for query in tqdm(queries_data, desc=f"Evaluating {model_name}"):
            q = query["query"]
            expected_chunk_id = query["expected_chunk_id"]
            document_id = query["document_id"]

            chunk_list = docid_to_chunks_list[document_id]
            chunk_ids = [chunk["id"] for chunk in chunk_list]
            chunk_texts = [chunk["content"] for chunk in chunk_list]

            # Encode query and each chunk
            query_vec = model.embed_query(q)
            chunk_vecs = model.embed_documents(chunk_texts)
            
            # Compute cosine similarities
            sims = np.dot(chunk_vecs, query_vec) / (
                np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
            )
            ranked_idx = np.argsort(sims)[::-1]  # Highest to lowest

            # Find position of expected_chunk_id
            try:
                rank = [chunk_ids[i] for i in ranked_idx].index(expected_chunk_id)
                rr = 1.0 / (rank + 1)
            except ValueError:
                rank = None
                rr = 0.0

            # Hit@K for each K
            for K in TOP_K_VALUES:
                if rank is not None and rank < K:
                    hit_counts[K] += 1

            # DCG: rel/log2(rank+2)
            # only expected_chunk_id is relevant, rel=1 for it, 0 elsewhere
            # DCG = 1/log2(rank+2) if found; IDCG always 1/log2(1+2)=1 
            if rank is not None:
                dcg = 1.0 / np.log2(rank + 2)
            else:
                dcg = 0.0
            idcg = 1.0 / np.log2(1 + 2)
            ndcg = dcg / idcg if idcg > 0 else 0

            reciprocal_ranks.append(rr)
            ndcgs.append(ndcg)

        end_time = time.time()
        latency = (end_time - start_time) / len(queries_data) if len(queries_data) > 0 else None

        total = len(queries_data)
        hit_rates = {f"HitRate@{k}": (hit_counts[k] / total if total else 0) for k in TOP_K_VALUES}
        mrr = sum(reciprocal_ranks) / total if total else 0
        mndcg = sum(ndcgs) / total if total else 0
        
        result_row = {
            "Model": model_name,
            **hit_rates,
            "MRR": mrr,
            "NDCG": mndcg,
            "Latency (s/query)": latency
        }
        results.append(result_row)

    df = pd.DataFrame(results)
    print("==== Evaluation Results ====")
    print(df.to_string(index=False))

    