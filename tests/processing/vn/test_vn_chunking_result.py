from src.data_loader.vn import VietnameseDataLoader
from src.processing.vn import VietnameseDocumentProcessor
from src.models.embeddings.vn_law_embedding import VNLawEmbedding
from collections import Counter 
from tqdm import tqdm 

import matplotlib.pyplot as plt
import numpy as np
import pytest

REQUIRED_CHUNK_KEYS = {"content", "metadata"}
REQUIRED_METADATA_KEYS = {"node_id", "header_name", "header_value", "header_index"}
VALID_LEVELS = {"phần", "chương", "mục", "tiểu_mục", "điều", "khoản", "điểm", ""} 

def validate_chunk(chunk: dict, expect_doc_metadata: bool = True) -> list[str]:
    errors = []
    if set(chunk.keys()) != REQUIRED_CHUNK_KEYS:
        errors.append(f"keys must be {REQUIRED_CHUNK_KEYS}, got {set(chunk.keys())}")

    content = chunk.get("content")
    if not isinstance(content, str):
        errors.append("content must be str")

    meta = chunk.get("metadata")
    if not isinstance(meta, dict):
        errors.append("metadata must be dict")
        return errors

    missing = REQUIRED_METADATA_KEYS - meta.keys()
    if missing:
        errors.append(f"metadata missing: {missing}")

    if meta.get("header_name") not in VALID_LEVELS:
        errors.append(f"invalid header_name: {meta.get('header_name')}")

    if expect_doc_metadata and "doc_id" not in meta:
        errors.append("metadata missing doc_id (needed for Qdrant)")

    return errors

def validate_document(processor, embedding_model, row, max_text_loss_ratio: float = 0.08) -> dict:
    doc_metadata = {
        "doc_id": str(row["id"]),
        "title": row["title"],
    }
    raw = row["content_html"]
    plain = processor.extract_text(raw)
    chunks = processor.process(raw, doc_metadata=doc_metadata)

    # Compute token count for each chunk individually (not as a list of all at once)
    token_counts = [embedding_model.count_tokens(chunk["content"]) for chunk in chunks]
    chunk_contents = [chunk["content"] for chunk in chunks]
    chunk_count = len(chunks)
    result = {
        "doc_id": doc_metadata["doc_id"],
        "title": doc_metadata["title"],
        "plain_len": len(plain),
        "chunk_count": chunk_count,
        "token_counts": token_counts,
        "chunk_contents": chunk_contents,
        "errors": [],
        "warnings": [],
    }

    if not plain.strip():
        result["warnings"].append("empty plain text after extract")
        return result

    if not chunks:
        result["errors"].append("no chunks produced")
        return result

    for i, ch in enumerate(chunks):
        result["errors"].extend(
            f"chunk[{i}]: {e}" for e in validate_chunk(ch)
        )

    joined = "\n".join(c["content"] for c in chunks)
    loss = 1 - (len(joined) / len(plain)) if plain else 0
    result["text_loss_ratio"] = loss
    if loss > max_text_loss_ratio and chunk_count < 100:
        result["errors"].append(
            f"text loss {loss:.1%} > {max_text_loss_ratio:.0%}"
        )

    # Warnings
    if any(c["metadata"]["node_id"] == "root" for c in chunks):
        root = next(c for c in chunks if c["metadata"]["node_id"] == "root")
        if len(root["content"]) > 2000:
            result["warnings"].append("root chunk very large")

    return result

def run_chunk_quality_check(
    batch_size: int = 100,
    offset: int = 0,
    max_fail_ratio: float = 0.05, 
) -> dict:

    loader = VietnameseDataLoader()
    processor = VietnameseDocumentProcessor()
    embedding_model = VNLawEmbedding()
    df = loader.load(batch_size=batch_size, offset=offset)
    
    results = []
    all_token_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating documents"):
        res = validate_document(processor, embedding_model, row)
        results.append(res)
        all_token_counts.extend(res.get("token_counts", []))

    stats = {}
    if all_token_counts:
        token_counts_np = np.array(all_token_counts)
        stats["mean"] = float(np.mean(token_counts_np))
        stats["median"] = float(np.median(token_counts_np))
        counts = Counter(token_counts_np)
        if counts:
            mode_val, _ = counts.most_common(1)[0]
            stats["mode"] = int(mode_val)
        else:
            stats["mode"] = None
        stats["min"] = int(np.min(token_counts_np))
        stats["max"] = int(np.max(token_counts_np))
        stats["std"] = float(np.std(token_counts_np))
        stats["percentile_95"] = float(np.percentile(token_counts_np, 95))
        stats["percentile_99"] = float(np.percentile(token_counts_np, 99))
        stats["pct_chunk_lt_512"] = float(np.sum(token_counts_np <= 512) / len(token_counts_np) * 100)
        stats["pct_chunk_ge_512"] = float(np.sum(token_counts_np > 512) / len(token_counts_np) * 100)
        
        print(f"===== Token count statistics for all chunks =====")
        print(f"Total chunks: {len(all_token_counts)}")
        print(f"Mean:    {stats['mean']:.2f}")
        print(f"Median:  {stats['median']:.2f}")
        print(f"Mode:    {stats['mode']}")
        print(f"Min:     {stats['min']}")
        print(f"Max:     {stats['max']}")
        print(f"Std:     {stats['std']:.2f}")
        print(f"95th percentile: {stats['percentile_95']:.2f}")
        print(f"99th percentile: {stats['percentile_99']:.2f}")
        print(f"% chunk <= 512:     {stats['pct_chunk_lt_512']:.2f}%")
        print(f"% chunk > 512:    {stats['pct_chunk_ge_512']:.2f}%")
        print(f"===============================================")

    failed = [r for r in results if r["errors"]]
    summary = {
        "total_docs": len(results),
        "failed_docs": len(failed),
        "fail_ratio": len(failed) / len(results) if results else 0,
        "avg_text_loss": sum(r.get("text_loss_ratio", 0) for r in results) / len(results),
        "failures_sample": failed[:10],
        "token_stats": stats
    }

    if summary["fail_ratio"] > max_fail_ratio:
        summary["passed"] = False
    else:
        summary["passed"] = True

    return summary

def plot_token_count_distribution(all_token_counts: list[int]):
    plt.figure(figsize=(10, 6))
    plt.hist(all_token_counts, bins=100, color='skyblue', edgecolor='black')
    plt.title("Chunk token count distribution (token < 1000)")
    plt.xlabel("Number of tokens in chunk")
    plt.ylabel("Number of chunks")
    plt.tight_layout()
    plt.savefig("token_count_distribution.jpg")
    
@pytest.mark.integration
def test_chunking_result():
    summary = run_chunk_quality_check(batch_size=18000, offset=0)    # 10% from total docs ~ 176k
    print(f"Summary: {summary}")
    assert summary["total_docs"] > 0
    assert summary["passed"] is True
    
    # # Print each failure sample
    # failures = summary.get("failures_sample", [])
    # for idx, fail in enumerate(failures, 1):
    #     print("-" * 100)
    #     print(f"[{idx}] doc_id={fail['doc_id']} | {fail['title']}")
    #     print(f"    chunks={fail['chunk_count']} | text_loss={fail.get('text_loss_ratio', 'n/a')}")
    #     for err in fail["errors"]:
    #         print(f"    - {err}")
    #     if fail.get("warnings"):
    #         print("    warnings:")
    #         for w in fail["warnings"]:
    #             print(f"    * {w}")