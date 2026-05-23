from .base import BaseDocumentProcessor
from bs4 import BeautifulSoup
import re

LEVEL_HIERARCHY = {
    'phần': 0, 'chương': 1, 'mục': 2,
    'tiểu_mục': 3, 'điều': 4, 'khoản': 5, 'điểm': 6
}

HEADER_PATTERNS = {
    'phần':     re.compile(r'^\s*phần\s+thứ\s+(\w+)', re.IGNORECASE| re.UNICODE),
    'chương':   re.compile(r'^\s*chương\s+([\divxlcdm]+[a-zđ]?)', re.IGNORECASE| re.UNICODE),
    'mục':      re.compile(r'^\s*mục\s+((?:x{0,3}(?:ix|iv|v?i{0,3})|\d+)[a-zđ]?)', re.IGNORECASE| re.UNICODE),
    'tiểu_mục': re.compile(r'^\s*tiểu\s+mục\s+(\d+[a-zđ]?)', re.IGNORECASE| re.UNICODE),
    'điều':     re.compile(r'^\s*điều\s+(\d+[a-zđ]?)', re.IGNORECASE| re.UNICODE),
    'khoản':    re.compile(r'^([1-9]\d*[a-zđ]?(?:\.\d+)?)[\.\-\)]\s*(?=\w)', re.IGNORECASE| re.UNICODE),
    'điểm': re.compile(fr'^([a-zđA-ZĐ]+)[\.\)\-]\s+', re.UNICODE),
}

class VietnameseDocumentProcessor(BaseDocumentProcessor):
    def extract_text(self, raw_content: str) -> str:
        if not raw_content:
            return ""
        if raw_content.strip().startswith("<"):
            return self._html_to_text(raw_content)
        return raw_content  

    def parse_structure(self, text: str) -> dict:
        return self._build_tree(text)

    def chunk(self, structure: dict, doc_metadata: dict = {}) -> list[dict]:
        """Convert structure → list of chunks with metadata"""
        if not structure:
            return []
        chunks = []
        exclude = {"content", "node_id"}
        for node_id, node in structure.items():
            content_formatted = self._format_content(node["content"])
            chunks.append({
                "content": content_formatted,
                "metadata": {
                    **doc_metadata,
                    "node_id": node_id,
                    **{k: v for k, v in node.items() if k not in exclude}
                }
            })
        return chunks

    def process(self, raw_content: str, doc_metadata: dict = {}) -> list[dict]:
        """Full ingestion pipeline"""
        text = self.extract_text(raw_content)
        structure = self.parse_structure(text)
        return self.chunk(structure, doc_metadata)
    
    # HELPER FUNCTIONS
    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        lines = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text:  
                lines.append(text)
                
        return "\n".join(lines)

    def _format_content(self, content: list[str]) -> str:
        return "\n".join(content)
    
    def _is_real_header(self, remainder: str) -> bool:
        if remainder == "":
            return True
        return bool(re.match(r'^[\s]*[.:\-]', remainder))
    
    def _resolve_diem_edge_case(self, header_value, remainder: str) -> tuple[str | None, str | None, str | None]:
        if header_value is None:
            return None, None, None
        stripped_remainder = remainder.strip(" .:-)")
        if re.match(r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})$', header_value):
            return "mục", header_value, stripped_remainder
        if header_value.isupper():
            return "tiểu_mục", header_value, stripped_remainder
        return "điểm", header_value, stripped_remainder
     
    def _classify_level(self, line: str) -> tuple[str | None, str | None]:
        """Returns: (header_name, header_value, remainder) or (None, None, None) if no match"""
        line = line.strip()
        for header_name, pattern in HEADER_PATTERNS.items():
            match = pattern.match(line)
            if match:
                remainder = line[match.end():]
                header_value = match.group(1)
                if not self._is_real_header(remainder) and header_name not in ("điểm", "khoản"):
                    return None, None, None
                if header_name == "điểm":
                    return self._resolve_diem_edge_case(header_value, remainder)
                return header_name, header_value, remainder.strip(" .:-")
        return None, None, None
        
    def _build_tree(self, text:str) -> dict|None:
        nodes: dict[str, dict] = {}
        stack = ["root"]
        
        nodes["root"] = {
            "node_id": "root",
            "header_name": "", 
            "header_value": "",
            "header_index": -1,
            "content": []
        }
        
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if line is None:
                continue
            
            header_name, header_value, remainder = self._classify_level(line)
            
            if header_name is None:
                nodes[stack[-1]]["content"].append(line)
                continue
            
            header_index = LEVEL_HIERARCHY[header_name]
            while len(stack) > 1 and nodes[stack[-1]]["header_index"] >= header_index:
                stack.pop()
            
            node_id = f"{stack[-1]}.{header_name}_{header_value}_{i}"
            nodes[node_id] = {
                "node_id": node_id,
                "header_name": header_name, 
                "header_value": header_value,
                "header_index": header_index,
                "content": [],
            }
            stack.append(node_id)
            if remainder: 
                nodes[node_id]['content'].append(remainder)
        return nodes


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

    # Schema từng chunk
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

    # Gợi ý thêm (warning, không fail cứng)
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
    from src.data_loader.vn import VietnameseDataLoader
    from src.processing.vn import VietnameseDocumentProcessor
    from tqdm import tqdm  

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

    

        
    