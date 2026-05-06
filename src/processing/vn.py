from .base import BaseDocumentProcessor
from bs4 import BeautifulSoup
import re
import time

LEVEL_HIERARCHY = ['phần', 'chương', 'mục', 'tiểu_mục', 'điều','khoản','điểm']
HEADER_PATTERNS = {
    'phần':      re.compile(r'^\s*phần\s+(thứ\s+)?([\divxlcdm]+)', re.UNICODE),
    'chương':    re.compile(r'^\s*chương\s+([\divxlcdm]+)', re.UNICODE),
    'mục':       re.compile(r'^\s*mục\s+(\d+)', re.UNICODE),
    'tiểu_mục':  re.compile(r'^\s*tiểu\s+mục\s+(\d+)', re.UNICODE),
    'điều':      re.compile(r'^\s*điều\s+(\d+)', re.UNICODE),
    'khoản': re.compile(r'^([1-9]\d?(?:\.\d+)?)[\.\-]\s*', re.UNICODE),
    'điểm':      re.compile(r'^([a-zđ])[/\.\)]\s+', re.UNICODE),
}

class VietnameseDocumentProcessor(BaseDocumentProcessor):
    def extract_text(self, raw_content: str) -> str:
        if not raw_content:
            return ""
        if raw_content.strip().startswith("<"):
            return self._html_to_text(raw_content)
        return raw_content  

    def parse_structure(self, text: str) -> dict:
        text = self._preprocess_text(text)
        
        lines = text.splitlines()
        preamble_lines = []
        for line in lines:
            normalized = line.strip().lower()
            found = any(HEADER_PATTERNS[l].match(normalized) for l in LEVEL_HIERARCHY)
            if found:
                break
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines).strip()
        
        return {
            "level": "root",
            "header": "",
            "content": preamble,
            "children": self._build_tree(text, 0)
        }

    def chunk(self, structure: dict, doc_metadata: dict = None) -> list[dict]:
        """Convert structure → list of chunks with metadata"""
        chunks = []
        self._collect_chunks(structure, ancestors=[], chunks=chunks, doc_metadata=doc_metadata or {})
        return chunks

    def process(self, raw_content: str, doc_metadata: dict = None) -> list[dict]:
        """Full ingestion pipeline"""
        # start = time.time()
        text = self.extract_text(raw_content)
        # end = time.time()
        # print(f"Time to extract text: {end - start:.2f}s")
        structure = self.parse_structure(text)
        return self.chunk(structure, doc_metadata)
    
    def _collect_chunks(self, node, ancestors, chunks, doc_metadata):
        if not node["children"]:
            breadcrumb = " > ".join(
                (a["header"] + (": " + a["content"].split("\n")[0] if a["content"] else ""))
                for a in ancestors if a["level"] != "root"
            )
            context = (breadcrumb + "\n" if breadcrumb else "") + node["header"]
            chunks.append({
                "content": (context + "\n" + node["content"]).strip(),
                "metadata": {
                    **doc_metadata, "level": node["level"], 
                    "header": node["header"],
                    "parent_header": ancestors[-1]["header"] if len(ancestors) > 1 else ""
                }
            })
        else:
            for child in node["children"]:
                self._collect_chunks(child, ancestors + [node], chunks, doc_metadata)   
        
    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    
    def _preprocess_text(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(':') and cleaned:
                cleaned[-1] = cleaned[-1] + stripped
            else:
                cleaned.append(stripped)
        return "\n".join(cleaned)
    
    def _build_tree(self, text: str, level_idx: int) -> list[dict]:
        if level_idx >= len(LEVEL_HIERARCHY):
            return []
        
        level = LEVEL_HIERARCHY[level_idx]
        segments = self._split_by_level(text, level)
        
        if len(segments) == 1 and segments[0][0] == "":
            return self._build_tree(text, level_idx + 1)
        
        nodes = []
        for header, content in segments:
            node = {
                "level": level,
                "header": header,
                "content": content,
                "children": self._build_tree(content, level_idx + 1)
            }
            nodes.append(node)
        return nodes
    
    def _split_by_level(self, text: str, level: str) -> list[tuple[str, str]]:
        """Split the text by the given level. Return a list of tuples, each tuple contains the level and the text content under that level"""
        
        pattern = HEADER_PATTERNS[level]
        lines = text.splitlines(keepends=True)
        
        # Find the indices of headers
        header_indices = []
        for i, line in enumerate(lines):
            if pattern.match(line.strip().lower()):
                header_indices.append(i)

        # No header -> there is only 1 segment
        if not header_indices:
            return [("", text)]
        
        segments = []
        
        for i, idx in enumerate(header_indices):
            full_header_line = lines[idx].strip()
            end = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            body = "".join(lines[idx + 1:end]).strip()

            if level in {"khoản", "điểm"}:
                parts = full_header_line.split(maxsplit=1)
                header_token = parts[0].rstrip(".-")
                inline_content = parts[1].strip() if len(parts) > 1 else ""
            else:
                m = HEADER_PATTERNS[level].match(full_header_line.lower())
                header_token = m.group(0).strip()
                inline_content = full_header_line[len(m.group(0)):].lstrip(':. ').strip()
            content = (inline_content + "\n" + body).strip() if inline_content else body
            segments.append((header_token, content))
        
        return segments

 

    

        
    