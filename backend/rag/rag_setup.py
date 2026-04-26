"""
RAG Setup Module
Handles document loading, splitting, embedding, and FAISS index creation
"""

import os
import numpy as np
import faiss
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FallbackEmbeddingModel:
    """Deterministic embedding fallback used when HF model loading fails."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(
        self,
        texts,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index, token in enumerate(text.lower().split()):
                slot = (sum(ord(char) for char in token) + index) % self.dimension
                vector[slot] += 1.0
            embeddings.append(vector)

        result = np.vstack(embeddings).astype(np.float32)
        return result if convert_to_numpy else result.tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


# ============================================================
# DOCUMENT LOADING & PREPROCESSING
# ============================================================

def load_text(file_path: str) -> str:
    """
    Load text content from a file.
    
    Args:
        file_path: Path to text file
    
    Returns:
        Text content as string
    
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Loaded {len(content)} characters from {file_path}")
        return content
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def load_multiple_texts(file_paths: List[str]) -> str:
    """
    Load and concatenate multiple text files.
    
    Args:
        file_paths: List of file paths
    
    Returns:
        Concatenated text content
    """
    all_content = []
    
    for file_path in file_paths:
        try:
            content = load_text(file_path)
            all_content.append(content)
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Skipping {file_path}: {e}")
    
    combined = "\n\n".join(all_content)
    logger.info(f"Loaded {len(file_paths)} files, total {len(combined)} characters")
    return combined


def load_pdf_texts(directory_path: str) -> str:
    """
    Load all text files from a directory (future: support PDFs).
    
    Args:
        directory_path: Path to directory containing text files
    
    Returns:
        Concatenated content from all text files
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Not a directory: {directory_path}")
    
    text_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith('.txt')
    ]
    
    if not text_files:
        logger.warning(f"No text files found in {directory_path}")
        return ""
    
    return load_multiple_texts(text_files)


# ============================================================
# TEXT CHUNKING & SPLITTING
# ============================================================

def split_text(text: str, chunk_size: int = 100, overlap: int = 0) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    
    Example:
        >>> text = "A" * 250
        >>> chunks = split_text(text, chunk_size=100, overlap=10)
        >>> len(chunks)
        3
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def split_text_by_sentences(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text by sentences (requires NLTK).
    
    Args:
        text: Text to split
        sentences_per_chunk: Number of sentences per chunk
    
    Returns:
        List of text chunks
    """
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Download tokenizer if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i : i + sentences_per_chunk])
            chunks.append(chunk)
        
        logger.info(f"Split into {len(chunks)} chunks by sentences")
        return chunks
    
    except ImportError:
        logger.warning("NLTK not available, falling back to character-based splitting")
        return split_text(text)


# ============================================================
# EMBEDDING & FAISS INDEX
# ============================================================

def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """
    Load embedding model from HuggingFace.
    
    Args:
        model_name: Model identifier (default: lightweight model)
    
    Returns:
        SentenceTransformer model instance
    
    Raises:
        Exception: If model cannot be loaded
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("✓ Model loaded successfully")
        return model
    except Exception as e:
        logger.warning(
            f"Falling back to deterministic local embeddings because loading "
            f"'{model_name}' failed: {e}"
        )
        return FallbackEmbeddingModel()


def create_embeddings(documents: List[str], 
                     model,
                     batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for documents.
    
    Args:
        documents: List of document chunks
        model: Embedding model
        batch_size: Batch size for processing
    
    Returns:
        Embedding matrix (N x D)
    """
    logger.info(f"Creating embeddings for {len(documents)} documents...")
    
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    logger.info(f"✓ Created embeddings with shape {embeddings.shape}")
    return embeddings


def create_faiss_index(embeddings: np.ndarray, 
                      index_type: str = "flat") -> faiss.Index:
    """
    Create FAISS index from embeddings.
    
    Args:
        embeddings: Embedding matrix (N x D)
        index_type: Type of index ('flat' | 'ivf' | 'hnsw')
    
    Returns:
        FAISS index ready for search
    
    Raises:
        ValueError: If embeddings are invalid
    """
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot create index with empty embeddings")
    
    dimension = embeddings.shape[1]
    embeddings = np.array(embeddings, dtype=np.float32)
    
    try:
        if index_type == "flat":
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings) // 10))
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        index.add(embeddings)
        logger.info(f"✓ Created FAISS index ({index_type}) with {index.ntotal} vectors")
        return index
    
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {e}")
        raise


def search_faiss_index(index: faiss.Index, 
                      query_embedding: np.ndarray,
                      k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for similar documents.
    
    Args:
        index: FAISS index
        query_embedding: Query embedding (1 x D)
        k: Number of results to return
    
    Returns:
        (distances, indices) - Distances and indices of k nearest neighbors
    """
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    distances, indices = index.search(query_embedding, k)
    return distances, indices


# ============================================================
# PIPELINE ORCHESTRATION
# ============================================================

class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    Manages loading, embedding, indexing, and retrieval.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        index_type: str = "flat"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model_name: HuggingFace model identifier
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_type: Type of FAISS index
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_type = index_type
        
        self.model = None
        self.documents = []
        self.embeddings = None
        self.index = None
        
        logger.info("RAGPipeline initialized")

    def load_documents(self, file_path: str) -> None:
        """Load and chunk documents from a file or a directory of `.txt` files."""
        if os.path.isdir(file_path):
            text = load_pdf_texts(file_path)
        else:
            text = load_text(file_path)

        self.documents = split_text(
            text,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        logger.info(f"Loaded {len(self.documents)} document chunks")

    def build_index(self) -> None:
        """Build embedding model and FAISS index."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        # Load model
        self.model = load_embedding_model(self.embedding_model_name)
        
        # Create embeddings
        self.embeddings = create_embeddings(self.documents, self.model)
        
        # Create FAISS index
        self.index = create_faiss_index(self.embeddings, index_type=self.index_type)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k similar documents.
        
        Args:
            query: Query string
            k: Number of results
        
        Returns:
            List of (document, distance) tuples
        """
        if self.model is None or self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode([query])
        distances, indices = search_faiss_index(self.index, query_embedding, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
        
        return results

    def setup(self, file_path: str) -> None:
        """
        Complete setup: load documents and build index.
        
        Args:
            file_path: Path to document file
        """
        self.load_documents(file_path)
        self.build_index()
        logger.info("✓ RAG pipeline ready for queries")


# ============================================================
# MAIN EXECUTION (For testing)
# ============================================================

if __name__ == "__main__":
    # Example usage
    rag = RAGPipeline()
    rag.setup("../../data/sample.txt")
    
    # Test retrieval
    results = rag.retrieve("What is backpropagation?", k=3)
    
    print("\n" + "="*60)
    print("Retrieval Results:")
    print("="*60)
    
    for doc, distance in results:
        print(f"\nDistance: {distance:.4f}")
        print(f"Document: {doc[:100]}...")
