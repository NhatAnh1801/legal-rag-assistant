from sentence_transformers import SentenceTransformer
import torch
from typing import List

class VNLawEmbedding():
    def __init__(self, batch_size: int=32):
        self.model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim = 128)
        self.model.max_seq_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device, dtype=torch.float16)
        self.batch_size = batch_size
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.model.encode(batch_texts)
            all_embeddings.extend(embeddings)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)
