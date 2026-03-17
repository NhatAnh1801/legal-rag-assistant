from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma

from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from src.models.embeddings.gte_multi_base import GTE
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
  
import os
import torch
import time
import requests

load_dotenv()

# DEFINE VARIABLES
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
COLAB_URL= ""

class RagController:
    def __init__(self):
        '''
            Init vector DB and LLM here
        '''
        # INIT MODELS
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        self.embedding_model = GTE()
        
        self.model = init_chat_model(
            "google_genai:gemini-flash-lite-latest",
            api_key=self.GEMINI_API_KEY
        )
        
        # INIT DATABASE
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        
        # INIT SMALL2BIG
        self.small2big_retriever = None
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size = PARENT_CHUNK_SIZE,
            length_function = len,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHILD_CHUNK_SIZE,
            length_function = len,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.small2big_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_db,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        self.store = LocalFileStore("./data/docstore")  
        
    def ingest_docs(self, docs: list, batch_size = 200):
        """
        Chunk documents using a hierarchical "small-to-big" approach for optimized retrieval.
        Note:
        - The maximum batch size supported by ChromaDB is 5461 documents.
        - The batch_size parameter is set to 50 to prevent overloading the vector database,
        - A pdf page full of text is about 3000 characters. -> ~4 parents, each parents have ~6 child => 24 child chunks each page
            -> The maximum capacity is 227 pages
            -> Set batch_size to 150-200 is the sweet spot
        This ensures stable ingestion and efficient use of system resources.
        """
        # Add data top retriever
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.small2big_retriever.add_documents(batch)
            
    def ingest_legal_docs(self):
        print("Sending ingestion request to Colab...")
        
        try:
            response = requests.post(
                f"{COLAB_URL}/ingest",
                timeout=3600  # 1 hour for large collections
            )
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                raise RuntimeError(data.get("message", "Unknown error"))
            
            for document in data["documents"]:
                jurisdiction_meta = document["jurisdiction"]
                domain_meta = document["domain"]
                source = document["source"]
                
                print(f"Ingesting: {jurisdiction_meta} -> {domain_meta}")
                
                # Wrap pages into LangChain Documents
                docs = [
                    Document(
                        page_content=page_text,
                        metadata={
                            "jurisdiction": jurisdiction_meta,
                            "domain": domain_meta,
                            "source": source,
                            "page": i
                        }
                    )
                    for i, page_text in enumerate(document["pages"])
                    if page_text.strip()  # skip empty pages
                ]
                
                clean_docs = filter_complex_metadata(docs)
                
                # Send to ChromaDB
                self.ingest_docs(clean_docs)
                print(f"✅ Done: {source}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot reach Colab. Is the tunnel still running?")
        except Exception as e:
            print(f"Ingestion failed: {e}")
   
    def ask(self, question, jurisdiction, domain, history=None, max_turns=5):
        @tool(response_format="content_and_artifact")
        def retrieve_doc(query:str) -> tuple:
            """
            Query documents from the user's question.
            Args:
                query: The user's question or search query.
            Returns:
                A tuple of (serialized text, list of Document objects).
            """
            try:
                self.small2big_retriever.vectorstore.search_kwargs = {
                    "filter": {
                        "$and": [
                            {"jurisdiction": {"$eq": jurisdiction}},
                            {"domain": {"$eq": domain}}
                        ]
                    }
                }
                
                retrieved_docs = self.small2big_retriever.invoke(query)
                serialized = "\n\n".join(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                print(f"Error during document retrieval: {e}")
                return "An error occurred while retrieving documents.", []
        
        system_prompt = f"""You are an expert Legal AI Assistant specializing in {jurisdiction} law, specifically within the {domain} domain.
        ## Context
        - You will receive a conversation history followed by the user's current legal question.
        - You MUST answer the question accurately based ONLY on the provided legal documents.
        - You have access to specialized tool_call actions, which should be leveraged to retrieve relevant documents and evidence necessary for answering user questions.
        
        ## Message Format
        You will receive messages in this order:
        - Previous conversation turns (for context)
        - The current user question (LAST message) -> this is what you need to answer    
        
        ## Constraint
        - Whenever you need to reference or search legal information in order to answer the user's question, you MUST use the retrieve_doc tool to retrieve the relevant documents first.
        - If the retrieved documents do not contain relevant information to answer the question, clearly state: "I couldn't find this information in the {jurisdiction}: {domain} documents."
        - If no documents are retrieved, clearly respond: "Sorry, I don't have information regarding legal matters in the {jurisdiction}: {domain} documents."
        - Do not fabricate laws, precedents, or rely on external knowledge outside the provided documents.
        
        ## Output format
        Strictly follow this output format:
        ```
        From my legal database, my answer is:
        [INSERT YOUR ANSWER HERE]
        ```
        """
        
        tools = [retrieve_doc]
        
        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt
        )
        
        messages = []
        if history:
            max_messages = max_turns * 2
            recent_history = history[-max_messages:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({"role": "user", "content": question})
        inputs = {"messages": messages}
        
        response = None
        for chunk in agent.stream(input=inputs, stream_mode="values"):
            response = chunk

        if response and "messages" in response:
            return response["messages"][-1].content
        return "No response"
        
    def test(self):
        print("\n" + "="*50)
        print("🚀 STARTING RAG CONTROLLER TEST")
        print("="*50)

        # 1. Define the test paths
        # Your ingest_legal_docs function looks for Path(base_directory).parent / "legal_docs"
        # We will pass the current working directory as the base.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        legal_docs_dir = Path(current_dir).parent / "legal_docs"

        print(f"📂 Looking for documents in: {legal_docs_dir}")
        
        if not legal_docs_dir.exists():
            print(f"⚠️  TEST HALTED: Directory '{legal_docs_dir}' does not exist.")
            print("Please create the following folder structure to run the test:")
            print("  legal_docs/")
            print("  └── Vietnam/")
            print("      └── Labor_Law.pdf")
            return

        # 2. Run the ingestion pipeline
        print("\n⚙️  STEP 1: Ingesting Documents...")
        try:
            self.ingest_legal_docs(base_directory=current_dir)
            print("✅ Ingestion complete.")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return

        # 3. Test the Agent Query
        print("\n🤖 STEP 2: Testing the LLM Agent...")
        
        # Test parameters that match the expected folder/file structure
        test_jurisdiction = "United States"
        test_domain = "AI Law"
        test_question = "What are the standard working hours according to this document?"
        
        print(f"   Jurisdiction: {test_jurisdiction}")
        print(f"   Domain:       {test_domain}")
        print(f"   Question:     {test_question}")
        print("\nThinking...")

        try:
            response = self.ask(
                question=test_question,
                jurisdiction=test_jurisdiction,
                domain=test_domain
            )
            
            print("\n" + "="*50)
            print("🎯 AGENT RESPONSE:")
            print("="*50)
            print(response)
            
        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            
    def test_orc(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        legal_docs_dir = Path(current_dir).parent / "legal_docs" / "Vietnam" / "Labor_Law.pdf"
        
        docs = self.load_and_process_docs(str(legal_docs_dir))
        print(f"\nLoaded {len(docs)} document(s) from {legal_docs_dir}")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(f"Metadata: {doc.metadata}")
            print("Content Preview:")
            print(doc.page_content[:10] + ("..." if len(doc.page_content) > 500 else ""))
        
        
        
            
if __name__ == "__main__":
    rag = RagController()
    rag.test()
    