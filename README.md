# Legal RAG assistant
A personal end-to-end Retrieval-Augmented Generation (RAG) system that helps look up and answer legal questions. Currently focused on Vietnamese legal documents, with plans to extend to other jurisdictions.

## Features
- **Multi-jurisdiction support**: Query legal documents from Vietnam and the United States, with an extensible structure for adding more countries.
- **Domain-specific filtering**: Narrow searches to specific legal fields such as AI Law, Labor Law, Enterprise Law, and Civil Procedure.
- **Smart retrieval (Small2Big)**: Uses a Parent-Child chunking strategy — searches on small chunks for precision, then returns the full parent context for comprehensive answers.
- **Conversation memory**: Maintains chat history so you can ask follow-up questions naturally.
- **Source citations**: Every document-based answer includes the exact PDF filename and page number for verification.
- **Bilingual support**: Handles legal questions in both Vietnamese and English, with plans to extend to more languages.

## Techstack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Agent Framework | LangGraph |
| Embedding Model | Alibaba-NLP/gte-multilingual-base |
| Reasoning LLM | Gemini 2.5 Flash Lite |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF (text-based PDFs), Docling + EasyOCR (scanned PDFs) |

## Future implementations
- **OCR processing speed**: EasyOCR takes ~120-130 seconds for 30 scanned pages on a T4 GPU (reduced from ~160s with float16). For larger documents (~80 pages), processing can take up to 480 seconds.
- **GPU constraints**: Development laptop (RTX 3060 6GB) cannot run both the GTE embedding model and EasyOCR simultaneously, requiring cloud GPU for OCR processing.
- **API quota**: Uses Gemini free tier (20 requests/day), which limits the number of questions that can be asked.
- **File format**: Currently supports PDF files only.

## Limitations
- Extend supported file formats beyond PDF
- Add evaluation framework for agent performance and parameter tuning
- Migrate processed documents to cloud database
- Further optimize EasyOCR processing time
- Extend to more jurisdictions and legal domains
