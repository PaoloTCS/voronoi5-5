import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from models.chunk import Chunk
from models.document import Document
import uuid

# Try to download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class TextProcessor:
    """
    Service for processing text documents, extracting chunks,
    identifying key points, and preprocessing for embedding.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        return text.strip()
    
    def extract_chunks_by_paragraphs(self, document: Document) -> List[Chunk]:
        """
        Extract chunks from a document by splitting it into paragraphs.
        
        Args:
            document: Document object to process
            
        Returns:
            List of Chunk objects
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', document.content)
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:  # Skip empty paragraphs
                continue
                
            # Create a unique ID for the chunk
            chunk_id = f"{document.id}_chunk_{str(uuid.uuid4())[:8]}"
            
            # Create a new chunk from the paragraph
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=paragraph,
                importance_rank=len(paragraphs) - i,  # Higher rank for earlier paragraphs
                key_point=self._extract_main_idea(paragraph),
                context_label=f"Paragraph {i+1}",
                start_char=document.content.find(paragraph),
                end_char=document.content.find(paragraph) + len(paragraph)
            )
            
            chunks.append(chunk)
        
        # Add chunks to document
        document.chunks = chunks
        
        return chunks
    
    def extract_chunks_by_sections(self, document: Document, section_markers: List[str] = None) -> List[Chunk]:
        """
        Extract chunks from a document by splitting it based on section headers.
        
        Args:
            document: Document object to process
            section_markers: List of regex patterns to identify section headers
            
        Returns:
            List of Chunk objects
        """
        if section_markers is None:
            # Default section markers - headers like "Chapter 1", "1. Introduction", "## Overview", etc.
            section_markers = [
                r'^#+\s+.*$',  # Markdown headers
                r'^\d+\.\s+.*$',  # Numbered sections
                r'^[A-Z][A-Za-z\s]+$',  # Capitalized headers
                r'^Chapter\s+\d+.*$',  # Chapter headers
                r'^Section\s+\d+.*$'  # Section headers
            ]
        
        # Combine markers into a single regex pattern
        combined_pattern = '|'.join(f'({marker})' for marker in section_markers)
        
        # Split text into lines
        lines = document.content.split('\n')
        
        sections = []
        current_section = []
        current_header = "Introduction"
        
        for line in lines:
            if re.match(combined_pattern, line.strip()):
                # Found a new section header
                if current_section:
                    # Save the previous section
                    sections.append((current_header, '\n'.join(current_section)))
                current_header = line.strip()
                current_section = []
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append((current_header, '\n'.join(current_section)))
        
        # Create chunks from sections
        chunks = []
        for i, (header, content) in enumerate(sections):
            if not content.strip():  # Skip empty sections
                continue
                
            # Create a unique ID for the chunk
            chunk_id = f"{document.id}_chunk_{str(uuid.uuid4())[:8]}"
            
            # Create a new chunk from the section
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=content,
                importance_rank=len(sections) - i,  # Higher rank for earlier sections
                key_point=self._extract_main_idea(content[:500]),  # Use first 500 chars for key point
                context_label=header,
                start_char=document.content.find(content),
                end_char=document.content.find(content) + len(content)
            )
            
            chunks.append(chunk)
        
        # Add chunks to document
        document.chunks = chunks
        
        return chunks
    
    def extract_chunks_by_fixed_size(self, document: Document, max_tokens: int = 200, overlap: int = 50) -> List[Chunk]:
        """
        Extract chunks from a document by splitting it into fixed-size segments.
        
        Args:
            document: Document object to process
            max_tokens: Maximum number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of Chunk objects
        """
        # Tokenize the document
        tokens = nltk.word_tokenize(document.content)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            if i + max_tokens > len(tokens):
                # Last chunk - use remaining tokens
                chunk_tokens = tokens[i:]
            else:
                chunk_tokens = tokens[i:i+max_tokens]
            
            # Convert tokens back to text
            chunk_text = ' '.join(chunk_tokens)
            
            # Create a unique ID for the chunk
            chunk_id = f"{document.id}_chunk_{str(uuid.uuid4())[:8]}"
            
            # Find position in original text
            start_pos = document.content.find(chunk_tokens[0])
            end_pos = start_pos + len(chunk_text)
            
            # Create chunk
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                importance_rank=len(tokens) // max_tokens - i // (max_tokens - overlap),  # Lower index, higher rank
                key_point=self._extract_main_idea(chunk_text),
                context_label=f"Chunk {i // (max_tokens - overlap) + 1}",
                start_char=start_pos,
                end_char=end_pos
            )
            
            chunks.append(chunk)
        
        # Add chunks to document
        document.chunks = chunks
        
        return chunks
    
    def _extract_main_idea(self, text: str, max_length: int = 100) -> str:
        """
        Extract the main idea or key point from a chunk of text.
        
        Args:
            text: Text to analyze
            max_length: Maximum length of the key point
            
        Returns:
            Extracted key point as a string
        """
        # Simple heuristic: use the first sentence as the key point
        sentences = nltk.sent_tokenize(text)
        
        if not sentences:
            return "No content"
        
        # Get first sentence
        first_sentence = sentences[0]
        
        # Truncate if too long
        if len(first_sentence) > max_length:
            return first_sentence[:max_length-3] + "..."
        
        return first_sentence
    
    def rank_chunks_by_importance(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rank chunks by their importance using simple heuristics.
        
        Args:
            chunks: List of Chunk objects to rank
            
        Returns:
            List of Chunk objects with updated importance_rank
        """
        # Simple heuristic: rank by position and length
        # Earlier chunks are usually more important (introduction, abstract)
        # Longer chunks may contain more information
        
        for i, chunk in enumerate(chunks):
            # Position score: earlier chunks get higher score
            position_score = len(chunks) - i
            
            # Length score: longer chunks get higher score
            length_score = len(chunk.content) / 1000  # Normalize by 1000 chars
            
            # Combine scores
            chunk.importance_rank = position_score + length_score
        
        # Sort by importance (descending)
        chunks.sort(key=lambda x: x.importance_rank, reverse=True)
        
        return chunks
    
    def analyze_chunks_for_coherence(self, chunks: List[Chunk]) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Analyze chunks for semantic coherence and transitions.
        
        Args:
            chunks: List of Chunk objects to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # This would typically use embeddings to compute semantic similarity
        # Here we use a simplified approach with shared words
        coherence_pairs = []
        
        for i in range(len(chunks)-1):
            for j in range(i+1, len(chunks)):
                # Get sets of words
                words_i = set(re.findall(r'\b\w+\b', chunks[i].content.lower()))
                words_j = set(re.findall(r'\b\w+\b', chunks[j].content.lower()))
                
                # Compute Jaccard similarity
                if words_i and words_j:
                    similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                else:
                    similarity = 0.0
                
                # Add if similarity is significant
                if similarity > 0.1:
                    coherence_pairs.append((i, j, similarity))
        
        # Sort by similarity (descending)
        coherence_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return {"coherence_pairs": coherence_pairs}
