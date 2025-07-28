import streamlit as st
from extractor import WebsiteExtractor
from summarizer import WebsiteSummarizer
from indexer import WebsiteIndexer
from chatbot import WebsiteChatbot
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config
import os
from dotenv import load_dotenv
import time
import re

# Load environment
load_dotenv()

# 🌐 Automatically verify Gemini backend key
try:
    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GEMINI_API_KEY
    )
    Config.set_gemini_key(Config.GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    ERROR_MSG = str(e)

# 🧠 Init state
def init_state():
    defaults = {
        'messages': [],
        'website_processed': False,
        'vector_db': None,
        'chatbot': None,
        'summary': '',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# Page config
st.set_page_config(
    page_title="🔒 Secure Website Chat",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.session_state["temperature"] = st.slider(
        "Response Creativity",
        0.0, 1.0, 0.7
    )

    st.session_state["max_pages"] = st.slider(
        "Max Pages to Crawl",
        1, 50, 10
    )

    st.session_state["chunk_size"] = st.selectbox(
        "Text Chunk Size",
        [512, 768, 1024, 1536],
        index=2
    )

    st.markdown("---")
    if GEMINI_AVAILABLE:
        st.success("✅ Gemini API Loaded")
    else:
        st.error(f"❌ Gemini API Error: {ERROR_MSG}")
        st.stop()

# Header
st.title("🛡️ Secure Website Chat")
st.markdown("Private, secure conversations with any website’s content. Just enter a URL below 👇")

# Input URL
with st.form("url_form"):
    url = st.text_input(
        "Paste Website URL (e.g. https://example.com)",
        placeholder="https://example.com"
    )
    process_btn = st.form_submit_button("🚀 Process Website")

# Processing
if process_btn and url:
    if not re.match(r"https?://", url):
        st.error("⚠️ Please include http:// or https:// in the URL.")
        st.stop()

    with st.status("🔎 Crawling & Indexing...", expanded=True) as status:
        try:
            # Step 1: Crawl
            extractor = WebsiteExtractor(url, max_pages=st.session_state.max_pages)
            pages = extractor.crawl_website()
            if not pages:
                st.error("❌ No content found on the website.")
                st.stop()
            st.success(f"✅ Extracted {len(pages)} pages.")

            # Step 2: Summarize
            summarizer = WebsiteSummarizer(llm)
            summary = summarizer.summarize_site(pages)
            st.session_state.summary = summary
            st.success("📝 Website summarized.")

            # Step 3: Index
            indexer = WebsiteIndexer(persist_dir="./chroma_db")
            chunks = indexer.index_website(pages, chunk_size=st.session_state.chunk_size)
            st.session_state.vector_db = indexer.vector_db
            st.session_state.chatbot = WebsiteChatbot(llm)
            st.session_state.chatbot.vector_db = indexer.vector_db
            st.session_state.website_processed = True
            st.success(f"🔐 Indexed {chunks} chunks.")

            status.update(label="✅ Done", state="complete")
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            status.update(label="❌ Failed", state="error")
            st.stop()

# Summary
if st.session_state.website_processed and st.session_state.summary:
    with st.expander("📄 Website Summary", expanded=True):
        st.markdown(st.session_state.summary)
        st.download_button("📥 Download Summary", st.session_state.summary, file_name="summary.md")

# Chat
if st.session_state.website_processed:
    st.subheader("💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Chat failed: {str(e)}")
