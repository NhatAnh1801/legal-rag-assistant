from sentence_transformers import SentenceTransformer
from src.models.embeddings.base import BaseEmbedding
from typing import List
from tqdm import tqdm
import time
import torch

class VNLawEmbedding(BaseEmbedding):
    def __init__(self, batch_size: int=64):
        self.model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim = 128)
        self.model.max_seq_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device, dtype=torch.float16)
        self.batch_size = batch_size
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches", total=((len(texts)-1)//self.batch_size + 1)):
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

    def _find_optimal_batch_size(self, test_length: int=512, max_test_batch: int=1024):
        """
        Benchmarks maximum GPU batch size and throughput.
        Returns:
            Optimal batch size for best performance on the current hardware.
        """
        sample_text = " ".join(["cat"] * test_length)     # 514 tokens
        print(f"Token count: {self.count_tokens(sample_text)}")
        
        current_batch = 1
        
        tested_batches = []
        throughputs = []
        
        TIMEOUT_SECONDS = 10.0 
        while current_batch <= max_test_batch:
            try:
                dummy_test = [sample_text] * current_batch
                
                warmup_start = time.time()
                _ = self.model.encode(dummy_test)
           
                if torch.cuda.is_available():
                    torch.cuda.synchronize() 
                
                warmup_time = time.time() - warmup_start
                
                if warmup_time > TIMEOUT_SECONDS:
                    print(f"Execution time is too long ({warmup_time:.2f}s). The GPU appears to be out of VRAM and is relying on system RAM, which will significantly degrade performance.")
                    break 
                
                num_runs = 3
                start_time = time.time()
                
                for _ in range(num_runs):
                    _ = self.model.encode(dummy_test)
                    
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                # 3. Calculate metrics
                avg_time = (end_time - start_time) / num_runs
                throughput = current_batch / avg_time 
                
                tested_batches.append(current_batch)
                throughputs.append(throughput)
                
                print(f"Passed: batch_size = {current_batch:<4} | Speed: {throughput:.2f} samples/sec")
                
                current_batch *= 2
                
            except torch.cuda.OutOfMemoryError:
                print(f"!!!OOM Error: Out of memory at batch_size = {current_batch}.")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"!!!OOM Error: Out of memory at batch_size = {current_batch}.")
                else:
                    raise e
                break
            finally:
                torch.cuda.empty_cache()
        
        if not tested_batches:
            print("!!!WARNING: The GPU cannot handle even batch_size = 1! Consider reducing max_seq_length.")
            return
            
        best_throughput = max(throughputs)
        best_perf_batch = tested_batches[throughputs.index(best_throughput)]
        
        self.batch_size = best_perf_batch
        return best_perf_batch