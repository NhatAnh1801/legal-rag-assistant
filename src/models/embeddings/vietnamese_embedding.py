from sentence_transformers import SentenceTransformer
from src.models.embeddings.base import BaseEmbedding
from typing import List
import torch

class VietnameseEmbedding(BaseEmbedding):
    def __init__(self, batch_size: int=32):
        self.model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
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
    
    def get_vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def count_tokens(self, text: str) -> int:
        encoding = self.model.tokenizer(text, return_tensors=None)
        return len(encoding["input_ids"])
    