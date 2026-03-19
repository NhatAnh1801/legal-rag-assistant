import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv

from src.models.embeddings.gte_multi_base import GTE
from src.prompt import*

import os
import requests
import time
import re       

load_dotenv()   

# DECLARE VARIABLES
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
CLOUDFLARE_URL= os.getenv("CLOUDFLARE_URL")

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
        
        self.llm_model = init_chat_model(
            "google_genai:gemini-flash-lite-latest",
            api_key=self.GEMINI_API_KEY
        )
        
        # Init Chroma as vector database
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory='./data/chromadb'    
        )
        
        # INIT SMALL2BIG
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
        
        fs = LocalFileStore("./data/docstore") 
        
        self.small2big_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_db,
            docstore=create_kv_docstore(fs),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
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
        if not docs:
            return
        
        # Check duplicate documents in the vector database
        existing = self.vector_db.get()
        ingested_keys = set(
            (m["source"], m["page"]) for m in existing["metadatas"]
        )
        new_docs = [
            d for d in docs
            if (d.metadata.get("source"), d.metadata.get("page")) not in ingested_keys
        ]
        
        skipped = len(docs) - len(new_docs)
        if skipped:
            print(f"-> [ingest_docs]: Skipping {skipped} already ingested pages")

        if not new_docs:
            print(f"-> [ingest_docs]: All documents already ingested, skipping...")
            return
        
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.small2big_retriever.add_documents(batch)
            
    def ingest_legal_docs(self):
        try:
            response = requests.get(
                f"{CLOUDFLARE_URL}/ingest",
                timeout=180
            )
            
            print(f"-> response: {response}")
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                raise RuntimeError(data.get("message", "Unknown error"))
            
            documents = data["documents"]
            print(f"-> [ingest_legal_docs]: Received {len(documents)} documents")
            
            all_docs = []
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
                    if page_text.strip() 
                ]
                all_docs.extend(docs)
                
            # Send to ChromaDB
            t0 = time.perf_counter()
            self.ingest_docs(all_docs)
            t1 = time.perf_counter()
            print(f"-> [ingest_legal_docs]: all docs are ingested in {t1 - t0: .4f}")
                
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot reach Colab. Is the tunnel still running?")
            raise
        except requests.exceptions.Timeout:
            print("-> [ingest_legal_docs]: Request timed out after 1 hour")
            raise
        except Exception as e:
            print(f"Ingestion failed: {e}")
            raise

    def get_retrieved_docs(self, jurisdiction: str, domain: str):
        # Verify if the jurisdiction and domain are valid
        if jurisdiction not in set(m["jurisdiction"] for m in self.vector_db.get()["metadatas"]):
            raise ValueError(f"Invalid jurisdiction: {jurisdiction}")
        if domain not in set(m["domain"] for m in self.vector_db.get()["metadatas"]):
            raise ValueError(f"Invalid domain: {domain}")
        
        @tool(response_format="content_and_artifact")
        def _retrieve_doc(query:str) -> tuple:
            """
                Query documents from the user's question.
                Args:
                    query: The user's question or search query.
                    jurisdiction: The jurisdiction to filter by.
                    domain: The domain to filter by.
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
            
        return _retrieve_doc
    
    def get_system_prompt(self, jurisdiction, domain):
        return system_prompt.format(jurisdiction=jurisdiction, domain=domain) + output_format
    
    def parse_response(self, response):
        try:
            clean = re.sub(r"```json|```", "", response).strip()
            if not clean.endswith("}"):
                clean += "}"
            return json.loads(clean)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
        
    def build_legal_agent(self, jurisdiction, domain):
        # build the RAG agent with the LLM, retrieval tools, and system prompt
        retrieve_doc = self.get_retrieved_docs(jurisdiction, domain)
        print(f"-> [build_legal_agent]: tool={retrieve_doc}")
        return create_agent(
            model=self.llm_model,
            tools=[retrieve_doc],
            system_prompt=self.get_system_prompt(jurisdiction, domain)
        )
        
    def ask(self, agent, question, history=None, max_turns=5):
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

        messages = response["messages"]
        # Debugging section
        human_msg = messages[0]
        final_ans = messages[-1]

        print(f"-> [ask]: Query: {human_msg.content}")

        has_tool_call = len(messages) >= 3  # Not docs based questions

        if has_tool_call:
            tool_call   = messages[1]
            tool_result = messages[2]

            print(f"-> [ask]: Retrieved {len(tool_result.artifact)} chunks | "
                f"Sources: {set(d.metadata['source'] for d in tool_result.artifact)} | "
                f"Pages: {[d.metadata['page'] for d in tool_result.artifact]}")

            t1 = tool_call.usage_metadata
            t2 = final_ans.usage_metadata
            print(f"-> [ask]: Tokens | "
                f"Call 1: {t1['total_tokens']} | "
                f"Call 2: {t2['total_tokens']} | "
                f"Total: {t1['total_tokens'] + t2['total_tokens']}")
        else:
            print(f"-> [ask]: No tool call (general questions)")
            t1 = final_ans.usage_metadata
            print(f"-> [ask]: Tokens | Total: {t1['total_tokens']}")
        
        if response and "messages" in response:
            return self.parse_response(final_ans.content)
        return None
    
    def _test(self):
        print("\n" + "="*50)
        print("🚀 STARTING RAG CONTROLLER TEST")
        print("="*50)
        
        # print("\n⚙️  STEP 1: Ingesting Documents...")
        # try:
        #     self.ingest_legal_docs()
        # except Exception as e:
        #     print(f"❌ Ingestion failed: {e}")
        #     return
        
        print("\n🤖 STEP 2: Testing the LLM Agent...")
        
        # Test parameters that match the expected folder/file structure
        test_jurisdiction = "Vietnam"
        #test_domain = "Enterprise Law"
        test_domain = "AI Law"
        test_question = "Khi nào hệ thống trí tuệ nhân tạo bị coi là rủi ro cao ?"
        #test_question = "Đất nước cuba nằm ở đâu?"
        try:
            agent = self.build_legal_agent(test_jurisdiction, test_domain)
            response = self.ask(
                question=test_question,
                agent=agent
            )
            
            print("\n" + "="*50)
            print("🎯 AGENT RESPONSE:")
            print("="*50)
            print(response)
            
        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            
    def _test_process_pdf(self):
        self.ingest_legal_docs()
        
def check_connection():
    response = requests.get(
        f"{CLOUDFLARE_URL}/health",
        timeout=60
    )
    print(response.text)
        
if __name__ == "__main__":
    rag = RagController()
    rag._test()
