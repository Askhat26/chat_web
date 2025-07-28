from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Optional
import os
import shutil
import logging
from config import Config


class WebsiteIndexer:
    def __init__(self, persist_dir: str = None):
        """
        Gemini-optimized website indexer with Chroma vector store.
        
        Args:
            persist_dir: Directory to persist vector store (default: None)
        """
        self.persist_dir = persist_dir
        self.vector_db = None
        self.logger = self._setup_logger()
        self.embeddings = self._initialize_embeddings()
        self.default_chunk_size = 1000  # Optimal for Gemini embeddings
        self.default_overlap = 200

    def _setup_logger(self):
        """Configure logging with Gemini-specific formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - GEMINI - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _initialize_embeddings(self):
        """Initialize Gemini embeddings with configuration validation."""
        config = Config.get_llm_config()
        if not config['api_key']:
            raise ValueError("Gemini API key not configured")
        
        return GoogleGenerativeAIEmbeddings(
            model=config['embedding_model'],
            google_api_key=config['api_key'],
            task_type="retrieval_document"  # Optimized for retrieval
        )

    def _count_tokens(self, text: str) -> int:
        """Gemini-optimized token approximation (word count + 20% buffer)."""
        return int(len(text.split()) * 1.2)  # Account for Gemini's tokenization

    def create_documents(self, pages: List[Dict[str, str]]) -> List[Document]:
        """
        Create LangChain Documents with Gemini-optimized metadata.
        
        Args:
            pages: List of {'url', 'title', 'content', optional 'summary'}
            
        Returns:
            List of Document objects with enhanced metadata
        """
        if not pages:
            raise ValueError("No pages provided for document creation")

        documents = []
        for page in pages:
            if not page.get('content'):
                self.logger.debug(f"Skipping empty page: {page.get('url')}")
                continue

            metadata = {
                'source': page['url'],
                'title': page.get('title', 'Untitled'),
                'summary': page.get('summary', ''),
                'language': 'en',
                'embedding_model': Config.GEMINI_EMBEDDING_MODEL
            }
            documents.append(Document(
                page_content=page['content'],
                metadata=metadata
            ))
        
        if not documents:
            raise ValueError("No valid documents created from pages")
            
        return documents

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Split documents into chunks optimized for Gemini embeddings.
        
        Args:
            documents: Documents to split
            chunk_size: Size in tokens (default: 1000)
            chunk_overlap: Overlap in tokens (default: 200)
            
        Returns:
            List of chunked Documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""]  # Sentence-aware
        )
        return splitter.split_documents(documents)

    def create_vector_store(
        self,
        documents: List[Document],
        collection_name: str = "website_content",
        overwrite: bool = False
    ) -> bool:
        """
        Create/update Chroma vector store with Gemini embeddings.
        
        Args:
            documents: Documents to index
            collection_name: Chroma collection name
            overwrite: Whether to overwrite existing data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if overwrite:
                self.clear_index()

            if self.persist_dir and os.path.exists(self.persist_dir):
                self.vector_db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                self.vector_db.add_documents(documents)
            else:
                self.vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_name=collection_name
                )

            if self.persist_dir:
                self.vector_db.persist()

            return True
        except Exception as e:
            self.logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
            return False

    def index_website(
        self,
        pages: List[Dict[str, str]],
        collection_name: str = "website_content",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> int:
        """
        Complete indexing pipeline optimized for Gemini.
        
        Args:
            pages: Website pages to index
            collection_name: Chroma collection name
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom overlap
            
        Returns:
            Number of chunks created, -1 on failure
        """
        try:
            documents = self.create_documents(pages)
            split_docs = self.split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            if not self.create_vector_store(split_docs, collection_name):
                raise RuntimeError("Vector store creation returned False")

            self.logger.info(f"Indexed {len(split_docs)} chunks")
            return len(split_docs)
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}", exc_info=True)
            return -1

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
        score_threshold: Optional[float] = None
    ):
        """Get configured retriever with Gemini-optimized defaults."""
        if not self.vector_db:
            raise RuntimeError("Vector store not initialized")

        search_kwargs = search_kwargs or {"k": 4}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return self.vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def clear_index(self) -> bool:
        """Completely reset the vector index."""
        try:
            if self.vector_db:
                self.vector_db.delete_collection()
                self.vector_db = None

            if self.persist_dir and os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir, exist_ok=True)

            return True
        except Exception as e:
            self.logger.error(f"Index clearing failed: {str(e)}", exc_info=True)
            return False

    def get_index_stats(self) -> Dict[str, any]:
        """Get statistics including Gemini embedding model info."""
        if not self.vector_db:
            return {"status": "not_initialized"}

        collection = self.vector_db._collection
        return {
            'document_count': collection.count() if collection else 0,
            'persisted': bool(self.persist_dir),
            'embedding_model': Config.GEMINI_EMBEDDING_MODEL,
            'chunk_size': self.default_chunk_size,
            'chunk_overlap': self.default_overlap
        }