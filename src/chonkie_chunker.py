from typing import List, Optional, Union

from beartype import beartype
from . import (RecursiveChunker, SDPMChunker, SemanticChunker,
                     SentenceChunker, TokenChunker)
from .embeddings.base import BaseEmbeddings
from symai import Expression, Symbol
from tokenizers import Tokenizer

CHUNKER_MAPPING = {
    "TokenChunker": TokenChunker,
    "SentenceChunker": SentenceChunker,
    "RecursiveChunker": RecursiveChunker,
    "SemanticChunker": SemanticChunker,
    "SDPMChunker": SDPMChunker,
}

@beartype
class ChonkieChunker(Expression):
    def __init__(
        self,
        tokenizer_name: Optional[str] = "gpt2",
        embedding_model_name: Optional[Union[str, BaseEmbeddings]] = "minishlab/potion-base-8M",
        **symai_kwargs,
    ):
        super().__init__(**symai_kwargs)
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model_name

    def forward(self, data: Symbol[Union[str, List[str]]], chunker_name: Optional[str] = "RecursiveChunker", **chunker_kwargs) -> Symbol[List[str]]:
        chunker = self._resolve_chunker(chunker_name, **chunker_kwargs)
        chunks = [ChonkieChunker.clean_text(chunk.text) for chunk in chunker(data.value)]
        return self._to_symbol(chunks)

    def _resolve_chunker(self, chunker_name: str, **chunker_kwargs) -> Union[TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker, SDPMChunker]:
        if chunker_name in ["TokenChunker", "SentenceChunker", "RecursiveChunker"]:
            tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            return CHUNKER_MAPPING[chunker_name](tokenizer, **chunker_kwargs)
        elif chunker_name in ["SemanticChunker", "SDPMChunker"]:
            return CHUNKER_MAPPING[chunker_name](embedding_model=self.embedding_model_name, **chunker_kwargs)
        else:
            raise ValueError(f"Chunker {chunker_name} not found. Available chunkers: {CHUNKER_MAPPING.keys()}.")

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text by removing problematic characters."""
        text = text.replace('\x00', '')                              # Remove null bytes (\x00)
        text = text.encode('utf-8', errors='ignore').decode('utf-8') # Replace invalid UTF-8 sequences
        return text
