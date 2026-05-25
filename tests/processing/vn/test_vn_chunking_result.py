from src.data_loader.vn import VietnameseDataLoader
from src.processing.vn import VietnameseDocumentProcessor
from tqdm import tqdm  

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

def validate_document(processor, row, max_text_loss_ratio: float = 0.08) -> dict:
    doc_metadata = {
        "doc_id": str(row["id"]),
        "title": row["title"],
    }
    raw = row["content_html"]
    plain = processor.extract_text(raw)
    chunks = processor.process(raw, doc_metadata=doc_metadata)
    chunk_count = len(chunks)
    result = {
        "doc_id": doc_metadata["doc_id"],
        "title": doc_metadata["title"],
        "plain_len": len(plain),
        "chunk_count": chunk_count,
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
    df = loader.load(batch_size=batch_size, offset=offset)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating documents"):
        results.append(validate_document(processor, row))

    failed = [r for r in results if r["errors"]]
    summary = {
        "total_docs": len(results),
        "failed_docs": len(failed),
        "fail_ratio": len(failed) / len(results) if results else 0,
        "avg_chunks": sum(r["chunk_count"] for r in results) / len(results),
        "avg_text_loss": sum(r.get("text_loss_ratio", 0) for r in results) / len(results),
        "failures_sample": failed[:10],
    }

    if summary["fail_ratio"] > max_fail_ratio:
        summary["passed"] = False
    else:
        summary["passed"] = True

    return summary

@pytest.mark.integration
def test_chunking_result():
    summary = run_chunk_quality_check(batch_size=18000, offset=0)    # 10% from total docs ~ 176k
    print(f"Summary: {summary}")
    assert summary["total_docs"] > 0
    assert summary["passed"] is True
    failures = summary["failures_sample"]
    # Print each failure sample
    for idx, fail in enumerate(failures, 1):
        print("-" * 100)
        print(f"[{idx}] doc_id={fail['doc_id']} | {fail['title']}")
        print(f"    chunks={fail['chunk_count']} | text_loss={fail.get('text_loss_ratio', 'n/a')}")
        for err in fail["errors"]:
            print(f"    - {err}")
        if fail.get("warnings"):
            print("    warnings:")
            for w in fail["warnings"]:
                print(f"    * {w}")