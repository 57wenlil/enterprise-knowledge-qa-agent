"""轻量 parent-child 文本切块。

Adapted conceptually from 用户提供的「高校智慧问答 RAG」的
`项目源码/core/document_processor.py` (parent/child chunks, stable parent metadata).
The original depends on heavyweight LangChain/OCR loaders; here we preserve
parent-child metadata and stable offsets with only the standard library.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    parent_id: str
    chunk_id: str
    start: int
    end: int
    text: str
    parent_text: str


def _window(text: str, size: int, overlap: int):
    """Yield nonempty overlapping windows, preferring natural paragraph breaks."""
    if not text:
        return
    pos = 0
    while pos < len(text):
        limit = min(len(text), pos + size)
        if limit < len(text):
            breaks = [text.rfind(ch, pos + size // 2, limit) for ch in ("\n", "。", "；")]
            cut = max(breaks)
            if cut > pos:
                limit = cut + 1
        yield pos, limit, text[pos:limit]
        if limit == len(text):
            break
        pos = max(pos + 1, limit - overlap)


def chunk_document(doc_id: str, content: str, parent_size: int = 1400,
                   child_size: int = 380, overlap: int = 40) -> list[Chunk]:
    if child_size < 40 or overlap < 0 or overlap >= child_size or parent_size < child_size:
        raise ValueError("invalid parent / child sizes")
    chunks: list[Chunk] = []
    for parent_idx, (parent_start, _parent_end, parent_text) in enumerate(
        _window(content, parent_size, overlap)
    ):
        parent_id = f"{doc_id}-p{parent_idx}"
        for _child_idx, (start, end, child_text) in enumerate(
            _window(parent_text, child_size, overlap)
        ):
            chunk_id = f"{doc_id}-{hashlib.sha256((str(parent_start + start) + child_text).encode()).hexdigest()[:12]}"
            chunks.append(Chunk(
                doc_id=doc_id, parent_id=parent_id, chunk_id=chunk_id,
                start=parent_start + start, end=parent_start + end,
                text=child_text, parent_text=parent_text,
            ))
    return chunks
