"""
Data Loaders for Language Model Perplexity Benchmark

Handles loading and preprocessing for:
- WikiText-103/WikiText-2 (standard LM benchmarks)
- C4 (diverse web text)
- PG-19 (long book sequences)

Provides chunked text for perplexity evaluation with sliding window.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from .config import DatasetConfig, PerplexityDataset, DATASET_CONFIGS

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TextChunk:
    """A chunk of text for perplexity evaluation."""
    
    text: str                              # Raw text
    input_ids: Optional[torch.Tensor] = None  # Tokenized [seq_len]
    attention_mask: Optional[torch.Tensor] = None
    
    # Metadata
    source_idx: int = 0                    # Index in original dataset
    chunk_idx: int = 0                     # Chunk index within source
    total_chunks: int = 1                  # Total chunks from source
    
    # For sliding window perplexity
    start_pos: int = 0                     # Start position in original doc
    end_pos: int = 0                       # End position in original doc


# =============================================================================
# Base Dataset Class
# =============================================================================

class PerplexityDataset(Dataset):
    """
    PyTorch Dataset for perplexity evaluation.
    
    Features:
    - Sliding window chunking with configurable stride
    - Pre-tokenization for efficiency
    - Handles variable-length documents
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        max_length: int = 4096,
        stride: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Load and preprocess data
        self.chunks: List[TextChunk] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and chunk the dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets required: pip install datasets")
        
        logger.info(f"Loading dataset: {self.config.hf_path}")
        
        # Load from HuggingFace
        if self.config.hf_subset:
            dataset = load_dataset(
                self.config.hf_path,
                self.config.hf_subset,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
            )
        else:
            dataset = load_dataset(
                self.config.hf_path,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
            )
        
        # Sample if needed
        if self.config.max_samples and len(dataset) > self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))
        
        logger.info(f"Processing {len(dataset)} documents...")
        
        # Process each document
        for doc_idx, item in enumerate(dataset):
            text = item[self.config.text_column]
            
            # Skip empty or very short texts
            if not text or len(text.strip()) < 10:
                continue
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = encoding.input_ids[0]  # [seq_len]
            
            # Skip if too short
            if len(input_ids) < self.config.min_length:
                continue
            
            # Chunk with sliding window
            chunks = self._chunk_document(
                text=text,
                input_ids=input_ids,
                doc_idx=doc_idx,
            )
            self.chunks.extend(chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks")
    
    def _chunk_document(
        self,
        text: str,
        input_ids: torch.Tensor,
        doc_idx: int,
    ) -> List[TextChunk]:
        """Split a document into overlapping chunks."""
        chunks = []
        seq_len = len(input_ids)
        
        if seq_len <= self.max_length:
            # Document fits in one chunk
            chunks.append(TextChunk(
                text=text,
                input_ids=input_ids,
                source_idx=doc_idx,
                chunk_idx=0,
                total_chunks=1,
                start_pos=0,
                end_pos=seq_len,
            ))
        else:
            # Sliding window chunking
            num_chunks = 0
            start = 0
            
            while start < seq_len:
                end = min(start + self.max_length, seq_len)
                
                chunk_ids = input_ids[start:end]
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    input_ids=chunk_ids,
                    source_idx=doc_idx,
                    chunk_idx=num_chunks,
                    start_pos=start,
                    end_pos=end,
                ))
                
                num_chunks += 1
                start += self.stride
                
                # Stop if remaining text is too short
                if seq_len - start < self.config.min_length // 2:
                    break
            
            # Update total_chunks
            for chunk in chunks:
                chunk.total_chunks = num_chunks
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        
        return {
            'input_ids': chunk.input_ids,
            'attention_mask': torch.ones_like(chunk.input_ids),
            'source_idx': chunk.source_idx,
            'chunk_idx': chunk.chunk_idx,
            'start_pos': chunk.start_pos,
            'end_pos': chunk.end_pos,
        }


# =============================================================================
# Specific Dataset Loaders
# =============================================================================

class WikiTextDataset(PerplexityDataset):
    """WikiText-103 or WikiText-2 dataset."""
    
    def _load_data(self) -> None:
        """Load WikiText with special handling for article structure."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets required: pip install datasets")
        
        logger.info(f"Loading WikiText: {self.config.hf_subset}")
        
        dataset = load_dataset(
            self.config.hf_path,
            self.config.hf_subset,
            split=self.config.split,
            cache_dir=self.config.cache_dir,
        )
        
        # WikiText is organized as one text per row
        # Concatenate into larger chunks for better context
        all_text = []
        current_article = []
        
        for item in dataset:
            text = item[self.config.text_column]
            
            # New article marker
            if text.startswith(" = ") and text.endswith(" = "):
                if current_article:
                    all_text.append(" ".join(current_article))
                current_article = [text]
            else:
                current_article.append(text)
        
        # Don't forget last article
        if current_article:
            all_text.append(" ".join(current_article))
        
        # Sample if needed
        if self.config.max_samples and len(all_text) > self.config.max_samples:
            all_text = all_text[:self.config.max_samples]
        
        logger.info(f"Processing {len(all_text)} articles...")
        
        # Process each article
        for doc_idx, text in enumerate(all_text):
            # Clean up
            text = text.strip()
            if len(text) < 10:
                continue
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = encoding.input_ids[0]
            
            if len(input_ids) < self.config.min_length:
                continue
            
            # Chunk
            chunks = self._chunk_document(text, input_ids, doc_idx)
            self.chunks.extend(chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from WikiText")


class C4Dataset(PerplexityDataset):
    """C4 dataset with streaming support for efficiency."""
    
    def _load_data(self) -> None:
        """Load C4 with streaming for memory efficiency."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets required: pip install datasets")
        
        logger.info(f"Loading C4 (streaming)...")
        
        # Use streaming for C4 since it's very large
        dataset = load_dataset(
            self.config.hf_path,
            self.config.hf_subset,
            split=self.config.split,
            streaming=True,
            cache_dir=self.config.cache_dir,
        )
        
        max_samples = self.config.max_samples or 1000
        
        logger.info(f"Processing up to {max_samples} samples...")
        
        for doc_idx, item in enumerate(dataset):
            if doc_idx >= max_samples:
                break
            
            text = item[self.config.text_column]
            
            if not text or len(text.strip()) < 10:
                continue
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = encoding.input_ids[0]
            
            if len(input_ids) < self.config.min_length:
                continue
            
            # Chunk
            chunks = self._chunk_document(text, input_ids, doc_idx)
            self.chunks.extend(chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from C4")


class PG19Dataset(PerplexityDataset):
    """PG-19 (books) dataset with long context handling."""
    
    def _load_data(self) -> None:
        """Load PG-19 with special handling for very long books."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets required: pip install datasets")
        
        logger.info(f"Loading PG-19 books...")
        
        dataset = load_dataset(
            self.config.hf_path,
            split=self.config.split,
            cache_dir=self.config.cache_dir,
        )
        
        # PG-19 has very long books, sample carefully
        max_samples = self.config.max_samples or 100
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        logger.info(f"Processing {len(dataset)} books...")
        
        for doc_idx, item in enumerate(dataset):
            text = item[self.config.text_column]
            
            if not text or len(text.strip()) < 100:
                continue
            
            # For very long books, sample from different parts
            # to get diverse context samples
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = encoding.input_ids[0]
            
            if len(input_ids) < self.config.min_length:
                continue
            
            # For very long books, use larger stride to cover more
            if len(input_ids) > self.max_length * 10:
                effective_stride = self.max_length * 2  # Less overlap for long books
            else:
                effective_stride = self.stride
            
            # Create chunks with adjusted stride
            old_stride = self.stride
            self.stride = effective_stride
            chunks = self._chunk_document(text, input_ids, doc_idx)
            self.stride = old_stride
            
            # Limit chunks per book to avoid one book dominating
            max_chunks_per_book = 50
            if len(chunks) > max_chunks_per_book:
                # Sample evenly from the book
                step = len(chunks) // max_chunks_per_book
                chunks = [chunks[i] for i in range(0, len(chunks), step)][:max_chunks_per_book]
            
            self.chunks.extend(chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from PG-19 books")


# =============================================================================
# Dataset Factory
# =============================================================================

def create_perplexity_dataset(
    config: DatasetConfig,
    tokenizer: Any,
    max_length: int = 4096,
    stride: Optional[int] = None,
) -> PerplexityDataset:
    """
    Create dataset based on configuration.
    
    Args:
        config: Dataset configuration
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        stride: Sliding window stride (None = max_length)
    
    Returns:
        PerplexityDataset instance
    """
    # Select appropriate dataset class
    if "wikitext" in config.hf_path.lower():
        dataset_class = WikiTextDataset
    elif "c4" in config.hf_path.lower():
        dataset_class = C4Dataset
    elif "pg19" in config.hf_path.lower():
        dataset_class = PG19Dataset
    else:
        dataset_class = PerplexityDataset
    
    return dataset_class(
        config=config,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )


def create_dataloader(
    dataset: PerplexityDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Create DataLoader for perplexity evaluation."""
    
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for variable-length sequences."""
        # Stack tensors
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'source_idx': [b['source_idx'] for b in batch],
            'chunk_idx': [b['chunk_idx'] for b in batch],
            'start_pos': [b['start_pos'] for b in batch],
            'end_pos': [b['end_pos'] for b in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_dataset_stats(dataset: PerplexityDataset) -> Dict[str, Any]:
    """Get statistics about the dataset."""
    if len(dataset) == 0:
        return {'num_chunks': 0}
    
    lengths = [len(chunk.input_ids) for chunk in dataset.chunks]
    
    return {
        'num_chunks': len(dataset),
        'total_tokens': sum(lengths),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'num_unique_sources': len(set(c.source_idx for c in dataset.chunks)),
    }


def load_all_datasets(
    configs: Dict[str, DatasetConfig],
    tokenizer: Any,
    max_length: int = 4096,
) -> Dict[str, PerplexityDataset]:
    """Load multiple datasets."""
    datasets = {}
    
    for name, config in configs.items():
        logger.info(f"Loading dataset: {name}")
        datasets[name] = create_perplexity_dataset(
            config=config,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        stats = get_dataset_stats(datasets[name])
        logger.info(f"  {name}: {stats['num_chunks']} chunks, {stats['total_tokens']} tokens")
    
    return datasets

