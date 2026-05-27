import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
import time

from tqdm import tqdm

class GTE():
    def __init__(self, batch_size: int=32):
        self.model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model = AutoModel.from_pretrained(
                self.model_name_or_path, 
                trust_remote_code=True, 
                dtype=torch.float16,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        
        self.batch_size = batch_size 
        
        self.model.eval()
        
    def _embedding(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(
            texts, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}   # Move the tensors to GPU 
        
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            
        embeddings = outputs.last_hidden_state[:, 0]    # CLS
        
        embeddings = F.normalize(embeddings, p=2, dim=1) # L2 Normalization
        
        return embeddings.tolist()
      
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        with tqdm(total=len(texts), desc="Embedding documents", unit="chunk") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                embeddings = self._embedding(batch_texts)
                all_embeddings.extend(embeddings)
                pbar.update(len(batch_texts))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_embeddings
        
    def embed_query(self, text: str) -> List[float]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._embedding([text])[0]
    
    def _find_optimal_batch_size(self, test_length: int=512, max_test_batch: int=1024):
            """
            Benchmarks maximum GPU batch size and throughput.
            Returns:
                Optimal batch size for best performance on the current hardware.
            """
            sample_text = "test " * test_length 
            
            current_batch = 1
            
            tested_batches = []
            throughputs = []
            
            TIMEOUT_SECONDS = 10.0  # Prevent GPU uses RAM when run out of VRAM 
            while current_batch <= max_test_batch:
                try:
                    dummy_test = [sample_text] * current_batch
                    
                    # 1. Warm-up: Run once to get the GPU engines running (crucial for accurate timing)
                    warmup_start = time.time()
                    _ = self._embedding(dummy_test)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize() # Force Python to wait until GPU finishes
                    
                    warmup_time = time.time() - warmup_start
                    
                    # Check if the program is using system RAM
                    if warmup_time > TIMEOUT_SECONDS:
                        print(f"Execution time is too long ({warmup_time:.2f}s). The GPU appears to be out of VRAM and is relying on system RAM, which will significantly degrade performance.")
                        break 
                    
                    # 2. Benchmark: Measure the time across a few runs to get a stable average
                    num_runs = 3
                    start_time = time.time()
                    
                    for _ in range(num_runs):
                        _ = self._embedding(dummy_test)
                        
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # 3. Calculate metrics
                    avg_time = (end_time - start_time) / num_runs
                    throughput = current_batch / avg_time # How many chunks processed per second
                    
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
                
            # Find the batch size that yielded the highest throughput
            best_throughput = max(throughputs)
            best_perf_batch = tested_batches[throughputs.index(best_throughput)]
            
            # Update the class attribute to use the absolute best batch size from now on
            self.batch_size = best_perf_batch
            return best_perf_batch

# if __name__ == "__main__":
#     from src.data_loader.vn import VietnameseDataLoader
#     from src.processing.vn import VietnameseDocumentProcessor
#     from src.models.embeddings.gte_multi_base import GTE
#     import matplotlib.pyplot as plt
#     import numpy as np

#     loader = VietnameseDataLoader()
#     processor = VietnameseDocumentProcessor()
#     gte = GTE(batch_size=32)

    # # Sample 500 docs for distribution
    # df = loader.load(batch_size=2000, offset=0)

    # all_lengths = []
    # doc_types = []
    # for _, row in tqdm(df.iterrows(), total=len(df)):
    #     chunks = processor.process(row["content_html"], doc_metadata={"doc_id": str(row["id"])})
    #     for chunk in chunks:
    #         tokens = gte.tokenizer.encode(chunk["content"])
    #         all_lengths.append(len(tokens))
    #         doc_types.append(row["loai_van_ban"])  

    # all_lengths = np.array(all_lengths)

    # print(f"Total chunks: {len(all_lengths)}")
    # print(f"Min:    {all_lengths.min()}")
    # print(f"Max:    {all_lengths.max()}")
    # print(f"Mean:   {all_lengths.mean():.0f}")
    # print(f"Median: {np.median(all_lengths):.0f}")
    # print(f"P90:    {np.percentile(all_lengths, 90):.0f}")
    # print(f"P95:    {np.percentile(all_lengths, 95):.0f}")
    # print(f"P99:    {np.percentile(all_lengths, 99):.0f}")
    # print(f">512:   {(all_lengths > 512).sum()} ({(all_lengths > 512).mean()*100:.1f}%)")
    # print(f">256:   {(all_lengths > 256).sum()} ({(all_lengths > 256).mean()*100:.1f}%)")
    
    # # Then after the main stats:
    # print("\n--- By doc type ---")
    # for loai in df["loai_van_ban"].unique():
    #     subset = [l for l, t in zip(all_lengths, doc_types) if t == loai]
    #     if subset:
    #         print(f"{loai:20} | count={len(subset):5} | mean={np.mean(subset):6.0f} | P95={np.percentile(subset, 95):6.0f} | max={max(subset):6}")

    # plt.figure(figsize=(12, 5))
    # plt.hist(all_lengths, bins=100, edgecolor='black')
    # plt.axvline(512, color='red', linestyle='--', label='512 tokens')
    # plt.axvline(256, color='orange', linestyle='--', label='256 tokens')
    # plt.xlabel("Token length")
    # plt.ylabel("Count")
    # plt.title("Chunk token length distribution")
    # plt.legend()
    # plt.show()
    
    # all_chunks = []
    # df = loader.load(batch_size=2000, offset=0)
    # for _, row in tqdm(df.iterrows(), total=len(df)):
    #     chunks = processor.process(row["content_html"], doc_metadata={"doc_id": str(row["id"])})
    #     all_chunks.extend(chunks)
        
    # print(f"Got {len(all_chunks)} chunks for benchmark")
    # gte = GTE(batch_size=32)
    # sample_texts = [c["content"] for c in all_chunks[:1000]]
    # for bs in [32, 64, 128, 256]:
    #     gte.batch_size = bs
    #     start = time.time()
    #     gte.embed_documents(sample_texts)
    #     elapsed = time.time() - start
    #     print(f"batch_size={bs}: {len(sample_texts)/elapsed:.1f} chunks/s")
        



        