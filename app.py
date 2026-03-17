import os
import shutil
import streamlit as st
import tempfile

from src.rag_engine import RagController

# --- 1. App Configuration ---
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# --- Define the Knowledge Hierarchy ---
# This dictionary maps the jurisdiction to its specific legal domains.
LEGAL_HIERARCHY = {
    "Vietnam": [
        "Tất cả (All)",
        "Trí tuệ nhân tạo (AI Law)", 
        "Lao động (Labor Law)", 
        "Doanh nghiệp (Enterprise Law)",
        "Dân sự (Civil Law)"
    ],
    "United States": [
        "All",
        "Corporate Law", 
        "Intellectual Property", 
        "Labor Law",
        "Constitutional Law"
    ],
    "United Kingdom": ["All", "Common Law", "Employment Law", "Company Law"],
    "European Union": ["All", "GDPR/Data Privacy", "AI Act", "Competition Law"]
}

# --- 2. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Legal AI Assistant. Please select your jurisdiction and legal domain to begin."}
    ]

if "rag_controller" not in st.session_state:
    st.session_state.rag_controller = RagController()

# --- 3. Sidebar: Database Info & Controls ---
with st.sidebar:
    st.header("⚖️ Legal Database Info")
    
    # 1st Dropdown: Jurisdiction Selector
    selected_jurisdiction = st.selectbox(
        "🌐 Select Jurisdiction",
        options=list(LEGAL_HIERARCHY.keys()),
        index=0,
    )
    
    # 2nd Dropdown: Domain Selector (Updates dynamically based on the 1st dropdown)
    selected_domain = st.selectbox(
        "📚 Select Legal Domain",
        options=LEGAL_HIERARCHY[selected_jurisdiction],
        index=0,
        help="Narrows the search down to a specific field of law."
    )
    
    # Store selections in session state
    st.session_state.jurisdiction = selected_jurisdiction
    st.session_state.domain = selected_domain

    st.info(
        f"**Targeting:**\n"
        f"{selected_domain}\n"
        f"from {selected_jurisdiction} law\n\n"
        "The assistant will filter the database to match these criteria."
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ready to answer questions regarding {st.session_state.jurisdiction} - {st.session_state.domain}."}
        ]
        st.rerun()

# --- 4. Main Chat Interface ---
st.title("🏛️ Legal Counsel RAG")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages[-50:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
# --- 5. Chat Input & Processing ---
question = st.chat_input(f"Ask a question about {selected_domain}...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(f"Reviewing {selected_jurisdiction} > {selected_domain} documents..."):
                history = st.session_state.messages[1:]
                
                try:
                    response = st.session_state.rag_controller.ask(
                        question=question, 
                        jurisdiction=st.session_state.jurisdiction,
                        domain=st.session_state.domain,
                        history=history
                    )
                except Exception as e:
                    response = f"An error occurred while querying the database: {str(e)}"
                
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
