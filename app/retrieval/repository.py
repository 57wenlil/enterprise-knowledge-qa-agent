"""Stable ID/version-aware source documents; immutable original Markdown."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import date
import json

from app.retrieval.chunker import Chunk, chunk_document


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    title: str
    text: str
    version: str
    version_date: str
    effective_from: str
    scope: str
    topic: str
    source: str
    claim_key: str
    claim_value: str
    authority: str | None
    chunks: tuple[Chunk, ...]

    def descriptor(self) -> dict:
        return {k: getattr(self, k) for k in (
            "doc_id", "title", "version", "version_date", "effective_from", "scope", "topic",
            "source", "claim_key", "claim_value", "authority"
        )}


class DocumentRepository:
    def __init__(self, data_dir: Path | str, as_of: str | None = None):
        self.data_dir = Path(data_dir).resolve()
        self.as_of = date.fromisoformat(as_of) if as_of else date.today()
        catalog = json.loads((self.data_dir / "catalog.json").read_text(encoding="utf-8"))
        self.documents: dict[str, KnowledgeDocument] = {}
        for item in catalog:
            doc_id = item["doc_id"]
            if doc_id in self.documents:
                raise ValueError(f"duplicate doc ID: {doc_id}")
            source_path = (self.data_dir / item["path"]).resolve()
            if not source_path.is_relative_to(self.data_dir):
                raise ValueError(f"unsafe document path: {item['path']}")
            if date.fromisoformat(item["effective_from"]) > self.as_of:
                continue  # avoid not-yet-effective documents
            original = source_path.read_text(encoding="utf-8")
            self.documents[doc_id] = KnowledgeDocument(
                **{key: item[key] for key in (
                    "doc_id", "title", "version", "version_date", "effective_from", "scope",
                    "topic", "source", "claim_key", "claim_value", "authority"
                )},
                text=original,
                chunks=tuple(chunk_document(doc_id, original)),
            )

    def get(self, doc_id: str) -> KnowledgeDocument:
        return self.documents[doc_id]

    def read(self, doc_id: str) -> dict:
        doc = self.get(doc_id)
        return {**doc.descriptor(), "content": doc.text, "chunk_ids": [c.chunk_id for c in doc.chunks]}


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
