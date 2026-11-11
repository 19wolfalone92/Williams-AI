"""Offline local multilingual embedding backend for Williams cognitive memory.

Intended for a locally installed multilingual sentence-transformers checkpoint.
No internet download, no cloud API, no remote code, no pickle model files.
Exact model-file SHA fingerprint binds memory vectors to their true encoder.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from .book_index import ROOT


class LocalSentenceEmbedding:
    def __init__(self, model_dir, *, device="cpu"):
        source=Path(model_dir).expanduser().resolve()
        if (not source.is_dir() or source.is_symlink() or
                source==ROOT or ROOT in source.parents):
            raise ValueError("LOCAL_EMBEDDING_DIRECTORY_OUTSIDE_GIT_REQUIRED")
        if device not in ("cpu","cuda"):
            raise ValueError("LOCAL_CPU_OR_CUDA_EMBEDDING_ONLY")
        weights=sorted(source.rglob("*.safetensors"))
        if not weights or any(p.is_symlink() for p in weights):
            raise ValueError("LOCAL_SAFETENSORS_EMBEDDING_WEIGHTS_REQUIRED")
        # No local pickled weights, no trust_remote_code execution.
        if any(source.rglob("*.bin")) or any(source.rglob("*.pt")) or any(source.rglob("*.pkl")):
            raise ValueError("UNSAFE_PICKLED_EMBEDDING_WEIGHTS_FORBIDDEN")
        digest=hashlib.sha256()
        allowed=set(weights)
        for path in sorted(p for p in source.rglob("*") if p.is_file() and
                           p.suffix.lower() in {
                               ".safetensors",".json",".txt",".model",".vocab"}):
            if path.is_symlink():
                raise ValueError("LOCAL_EMBEDDING_SYMLINK_NOT_ALLOWED")
            relative=str(path.relative_to(source)).encode()
            digest.update(len(relative).to_bytes(4,"big")+relative)
            with path.open("rb") as stream:
                while True:
                    block=stream.read(1<<20)
                    if not block:break
                    digest.update(block)
        self.model_id="local-st-"+digest.hexdigest()[:32]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("INSTALL_OPTIONAL_SENTENCE_TRANSFORMERS_LOCALLY") from exc
        self.model=SentenceTransformer(
            str(source),device=device,trust_remote_code=False,
            local_files_only=True)
        self.dimension=int(self.model.get_sentence_embedding_dimension())
        if not 2<=self.dimension<=1024:
            raise ValueError("UNSUPPORTED_EMBEDDING_DIMENSIONS")

    def __call__(self,text):
        if not isinstance(text,str) or not 2<=len(text)<=6000:
            raise ValueError("BOUNDED_LOCAL_SEMANTIC_TEXT_REQUIRED")
        encoded=self.model.encode(text,normalize_embeddings=True,
                                  convert_to_numpy=True,show_progress_bar=False)
        values=encoded.tolist()
        if not isinstance(values,list) or len(values)!=self.dimension:
            raise ValueError("INCOMPATIBLE_LOCAL_SENTENCE_ENCODER")
        return values
