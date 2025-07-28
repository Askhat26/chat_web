from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Optional
from config import Config
import os
import logging
from datetime import datetime

class WebsiteChatbot:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Gemini-optimized chatbot for website content interaction.
        
        Args:
            llm: Initialized ChatGoogleGenerativeAI instance
        """
        self.llm = llm
        self.vector_db = None
        self.logger = self._setup_logger()
        self.embeddings = self._initialize_embeddings()
        self.default_chunk_size = 1000  # Optimal for Gemini
        self.default_overlap = 200

    def _setup_logger(self):
        """Configure Gemini-specific logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - GEMINI-CHATBOT - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _initialize_embeddings(self):
        """Initialize and validate Gemini embeddings."""
        config = Config.get_llm_config()
        return GoogleGenerativeAIEmbeddings(
            model=config['embedding_model'],
            google_api_key=config['api_key'],
            task_type="retrieval_document"  # Optimized for QA
        )

    def _count_tokens(self, text: str) -> int:
        """Enhanced token approximation for Gemini."""
        return int(len(text.split()) * 1.2)  # 20% buffer for Gemini tokenization

    def prepare_vector_db(
        self,
        pages: List[Dict[str, str]],
        persist_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> int:
        """
        Prepare vector database with Gemini-optimized settings.
        
        Args:
            pages: List of {'url', 'title', 'content', optional 'summary'}
            persist_dir: Directory to persist vector store
            chunk_size: Custom chunk size (default: 1000)
            chunk_overlap: Custom overlap (default: 200)
            
        Returns:
            Number of chunks created
        """
        try:
            documents = []
            seen_urls = set()
            
            for page in pages:
                if not page.get('content') or page['url'] in seen_urls:
                    continue
                    
                seen_urls.add(page['url'])
                documents.append(Document(
                    page_content=page['content'],
                    metadata={
                        'source': page['url'],
                        'title': page.get('title', 'Untitled'),
                        'summary': page.get('summary', ''),
                        'processed_at': datetime.now().isoformat(),
                        'embedding_model': Config.GEMINI_EMBEDDING_MODEL
                    }
                ))

            if not documents:
                raise ValueError("No valid documents created from pages")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self.default_chunk_size,
                chunk_overlap=chunk_overlap or self.default_overlap,
                length_function=self._count_tokens,
                separators=["\n\n", "\n", "(?<=\\. )", " ", ""]  # Sentence-aware
            )
            split_docs = splitter.split_documents(documents)

            self.vector_db = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=persist_dir,
                collection_name="website_content"
            )

            if persist_dir:
                self.vector_db.persist()
                self.logger.info(f"Persisted vector store to {persist_dir}")

            return len(split_docs)

        except Exception as e:
            self.logger.error(f"Vector DB preparation failed: {str(e)}", exc_info=True)
            raise

    def get_qa_chain(
        self,
        chain_type: str = "stuff",
        search_kwargs: Optional[Dict] = None
    ) -> RetrievalQA:
        """
        Create a Gemini-optimized QA chain.
        
        Args:
            chain_type: "stuff", "map_reduce", "refine", or "map_rerank"
            search_kwargs: Custom search parameters
            
        Returns:
            Configured RetrievalQA chain
        """
        if not self.vector_db:
            raise RuntimeError("Vector DB not initialized. Call prepare_vector_db() first.")

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_kwargs=search_kwargs or {
                    "k": 4,  # Optimal for Gemini's context window
                    "score_threshold": 0.7
                }
            ),
            chain_type=chain_type,
            return_source_documents=True,
            verbose=Config.DEBUG_MODE
        )

    def chat(
        self,
        question: str,
        qa_chain: Optional[RetrievalQA] = None,
        max_source_chars: int = 1000
    ) -> Dict[str, any]:
        """
        Enhanced chat interaction with source validation.
        
        Args:
            question: User's query
            qa_chain: Pre-configured QA chain (optional)
            max_source_chars: Max characters to show per source
            
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "timestamp": str
            }
        """
        try:
            qa_chain = qa_chain or self.get_qa_chain()
            result = qa_chain({"query": question})
            
            # Process sources
            sources = []
            for doc in result["source_documents"]:
                content = doc.page_content[:max_source_chars]
                if len(doc.page_content) > max_source_chars:
                    content += "..."
                    
                sources.append({
                    "url": doc.metadata.get('source', ''),
                    "title": doc.metadata.get('title', 'Untitled'),
                    "content": content,
                    "relevance_score": doc.metadata.get('score', 0)
                })

            return {
                "answer": result["result"],
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Chat failed: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing your request: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }

    def load_vector_db(
        self,
        persist_dir: str,
        collection_name: str = "website_content"
    ) -> bool:
        """
        Safely load persisted vector database.
        
        Args:
            persist_dir: Directory containing persisted data
            collection_name: Chroma collection name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(persist_dir):
                raise FileNotFoundError(f"Directory not found: {persist_dir}")

            self.vector_db = Chroma(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            self.logger.info(f"Loaded vector DB from {persist_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector DB: {str(e)}", exc_info=True)
            return False

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict] = None
    ):
        """Get configured retriever with validation."""
        if not self.vector_db:
            raise RuntimeError("Vector DB not initialized")
            
        return self.vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 4}
        )