from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List, Dict, Optional, Union
import os
import shutil
import logging
from config import Config


class WebsiteIndexer:
    def __init__(self, persist_dir: str = None):
        """
        Enhanced website indexer with support for multiple embedding models.
        
        Args:
            persist_dir: Directory to persist vector store (optional)
        """
        self.persist_dir = persist_dir
        self.vector_db = None
        self.logger = self._setup_logger()
        self.embeddings = self._initialize_embeddings()

    def _setup_logger(self):
        """Configure logging for the indexer."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _initialize_embeddings(self):
        """Initialize embeddings based on configuration."""
        config = Config.get_llm_config()
        
        if config['type'] == 'openai':
            from langchain.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=config['embedding_model'],
                openai_api_key=config['api_key']
            )
        else:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=config['embedding_model'],
                base_url=config.get('base_url', 'http://localhost:11434')
            )

    def _count_tokens(self, text: str) -> int:
        """Approximate token counting for text splitting."""
        return len(text.split())  # Simple word count approximation

    def create_documents(self, pages: List[Dict[str, str]]) -> List[Document]:
        """
        Convert crawled pages into LangChain Document objects with enhanced metadata.
        
        Args:
            pages: List of page dictionaries with url, title, content, and optional summary
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        for page in pages:
            if not page.get('content'):
                self.logger.warning(f"Skipping empty page: {page.get('url')}")
                continue

            metadata = {
                'source': page['url'],
                'title': page.get('title', ''),
                'summary': page.get('summary', ''),
                'language': 'en'  # Can be dynamically detected
            }
            documents.append(Document(
                page_content=page['content'],
                metadata=metadata
            ))
        return documents

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function = None
    ) -> List[Document]:
        """
        Split documents into chunks with improved handling of different content types.
        
        Args:
            documents: List of Documents to split
            chunk_size: Target size of chunks (in tokens)
            chunk_overlap: Overlap between chunks
            length_function: Optional custom length function
            
        Returns:
            List of chunked Documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function or self._count_tokens,
            separators=["\n\n", "\n", " ", ""]  # Improved splitting logic
        )
        return splitter.split_documents(documents)

    def create_vector_store(
        self,
        documents: List[Document],
        collection_name: str = "website_content",
        overwrite: bool = False
    ) -> bool:
        """
        Create or update a Chroma vector store with enhanced error handling.
        
        Args:
            documents: Documents to index
            collection_name: Name of the collection
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if overwrite and self.persist_dir:
                self.clear_index()

            if self.persist_dir and os.path.exists(self.persist_dir):
                # Load existing store
                self.vector_db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                self.vector_db.add_documents(documents)
            else:
                # Create new store
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
            self.logger.error(f"Failed to create vector store: {str(e)}")
            return False

    def index_website(
        self,
        pages: List[Dict[str, str]],
        collection_name: str = "website_content",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> int:
        """
        Complete indexing pipeline with improved error handling.
        
        Args:
            pages: List of page dictionaries
            collection_name: Chroma collection name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks created (-1 if failed)
        """
        try:
            documents = self.create_documents(pages)
            if not documents:
                raise ValueError("No valid documents to index")

            split_docs = self.split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            if not self.create_vector_store(split_docs, collection_name):
                raise RuntimeError("Failed to create vector store")

            return len(split_docs)

        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}")
            return -1

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 4},
        score_threshold: Optional[float] = None
    ):
        """
        Get a retriever with enhanced search options.
        
        Args:
            search_type: Type of search (similarity, mmr, etc.)
            search_kwargs: Search parameters
            score_threshold: Minimum similarity score
            
        Returns:
            Configured retriever object
        """
        if not self.vector_db:
            raise RuntimeError("Vector DB not initialized")

        search_kwargs = search_kwargs.copy()
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return self.vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def get_vector_db(self):
        """Get the current vector database instance."""
        return self.vector_db

    def clear_index(self):
        """Completely clear the vector index with proper cleanup."""
        try:
            if self.vector_db:
                self.vector_db.delete_collection()
                self.vector_db = None

            if self.persist_dir and os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir, exist_ok=True)

            return True
        except Exception as e:
            self.logger.error(f"Failed to clear index: {str(e)}")
            return False

    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        if not self.vector_db:
            return {}
        
        collection = self.vector_db._collection
        return {
            'document_count': collection.count() if collection else 0,
            'persisted': bool(self.persist_dir),
            'embedding_model': str(self.embeddings)
        }