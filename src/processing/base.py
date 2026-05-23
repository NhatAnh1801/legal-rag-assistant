from abc import ABC, abstractmethod

class BaseDocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, raw_content: str) -> str:
        """Convert raw content (HTML/PDF/Markdown) → plain text"""
        pass

    @abstractmethod
    def parse_structure(self, text: str) -> dict:
        """Extract hierarchy: chapter/article/clause"""
        pass

    @abstractmethod
    def chunk(self, structure: dict, doc_metadata: dict = {}) -> list[dict]:
        """Convert structure → list of chunks with metadata"""
        pass

    def process(self, raw_content: str, doc_metadata: dict = {}) -> list[dict]:
        """Full Pipeline"""
        pass