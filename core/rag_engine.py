"""
Standards-grounded RAG engine for ManufacturingRAG-QA.

Pipeline:
    PDF standards
        ↓
    Page-aware text extraction
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    FAISS semantic retrieval
        +
    BM25 lexical retrieval
        ↓
    Reciprocal Rank Fusion (RRF)
        ↓
    Defect-aware reranking
        ↓
    Standards-grounded answers / audit evidence

Primary evidence source:
    Manufacturer / technical PDF documents

Fallback / compatibility source:
    core.standards_kb.py
"""

import os
import re
from typing import Any, Dict, List, Optional

import fitz
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

from core.standards_kb import (
    get_all_standards,
    get_standard_by_defect_type,
    search_standards_kb,
)


class StandardsRAGEngine:
    """
    Hybrid standards RAG engine.

    Retrieval:
        FAISS semantic search
        +
        BM25 lexical search
        +
        Reciprocal Rank Fusion

    Answering:
        Intent detection
        +
        Defect detection
        +
        Strict defect-aware evidence filtering
        +
        Structured standards KB
    """

    # =============================================================
    # DEFECT ALIASES
    # =============================================================

    DEFECT_ALIASES = {
        "solder bridging": [
            "solder bridging",
            "solder bridge",
            "solder bridges",
            "solder short",
            "solder shorts",
            "bridge between solder",
            "bridging",
        ],
        "solder balls": [
            "solder balls",
            "solder ball",
        ],
        "tombstoning": [
            "tombstoning",
            "tombstone",
        ],
        "missing component": [
            "missing component",
            "missing components",
            "component missing",
        ],
        "component misalignment": [
            "component misalignment",
            "component misaligned",
            "misaligned component",
            "component offset",
            "side overhang",
            "overhang",
        ],
        "insufficient solder": [
            "insufficient solder",
            "insufficient soldering",
            "low solder",
            "not enough solder",
        ],
        "poor wetting": [
            "poor wetting",
            "non-wetting",
            "non wetting",
            "dewetting",
        ],
        "cold solder joint": [
            "cold solder joint",
            "cold solder",
        ],
        "voiding": [
            "solder voiding",
            "solder void",
            "voiding",
        ],
    }

    # Defect names which should NOT be considered interchangeable.
    UNRELATED_DEFECTS = [
        "side overhang",
        "component misalignment",
        "tombstoning",
        "solder ball",
        "solder balls",
        "missing component",
        "insufficient solder",
        "poor wetting",
        "cold solder",
        "voiding",
    ]

    # =============================================================
    # INITIALIZATION
    # =============================================================

    def __init__(
        self,
        standards_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        rrf_k: int = 60,
    ):

        # ---------------------------------------------------------
        # Project paths
        # ---------------------------------------------------------

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
            )
        )

        if standards_dir is None:
            standards_dir = os.path.join(
                project_root,
                "data",
                "standards",
            )

        self.standards_dir = os.path.abspath(
            standards_dir
        )

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------

        if embedding_model is None:
            embedding_model = os.environ.get(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )

        self.embedding_model_name = embedding_model

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------

        self.embedding_model = None

        self.chunks: List[Dict[str, Any]] = []

        self.chunk_texts: List[str] = []

        self.faiss_index = None

        self.bm25 = None

        self.embedding_dimension: Optional[int] = None

        # ---------------------------------------------------------
        # Initialize
        # ---------------------------------------------------------

        self._load_embedding_model()

        self._build_knowledge_base()

    # =============================================================
    # EMBEDDING MODEL
    # =============================================================

    def _load_embedding_model(self):

        print("[RAG] Loading embedding model:")
        print(f"      {self.embedding_model_name}")

        hf_home = os.environ.get("HF_HOME")

        if hf_home:
            print("[RAG] Hugging Face cache:")
            print(f"      {hf_home}")

        # Prefer local cached models.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_path = self.embedding_model_name

        if hf_home:

            model_folder = (
                "models--"
                + self.embedding_model_name.replace("/", "--")
            )

            snapshots_dir = os.path.join(
                hf_home,
                "hub",
                model_folder,
                "snapshots",
            )

            if os.path.isdir(snapshots_dir):

                snapshots = [
                    os.path.join(
                        snapshots_dir,
                        name,
                    )
                    for name in os.listdir(
                        snapshots_dir
                    )
                    if os.path.isdir(
                        os.path.join(
                            snapshots_dir,
                            name,
                        )
                    )
                ]

                if snapshots:
                    model_path = snapshots[0]

                    print(
                        f"[RAG] Using local snapshot: "
                        f"{model_path}"
                    )

        try:

            self.embedding_model = (
                SentenceTransformer(model_path)
            )

        except Exception:

            self.embedding_model = (
                SentenceTransformer(
                    self.embedding_model_name
                )
            )

        # ---------------------------------------------------------
        # Verify embedding dimension
        # ---------------------------------------------------------

        test_embedding = (
            self.embedding_model.encode(
                [
                    "PCB manufacturing quality inspection"
                ],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        test_embedding = np.asarray(
            test_embedding
        )

        if test_embedding.ndim == 1:

            test_embedding = (
                test_embedding.reshape(
                    1,
                    -1,
                )
            )

        self.embedding_dimension = int(
            test_embedding.shape[1]
        )

        print("[RAG] Embedding model ready:")
        print(
            f"      Dimension: "
            f"{self.embedding_dimension}"
        )

    # =============================================================
    # KNOWLEDGE BASE
    # =============================================================

    def _build_knowledge_base(self):

        os.makedirs(
            self.standards_dir,
            exist_ok=True,
        )

        pdf_files = [
            os.path.join(
                self.standards_dir,
                filename,
            )
            for filename in os.listdir(
                self.standards_dir
            )
            if filename.lower().endswith(".pdf")
        ]

        # ---------------------------------------------------------
        # Fallback directory
        # ---------------------------------------------------------

        if not pdf_files:

            alt_dir = (
                "D:/ManufacturingRAG-QA/data/standards"
            )

            if os.path.exists(alt_dir):

                import shutil

                for filename in os.listdir(
                    alt_dir
                ):

                    if filename.lower().endswith(".pdf"):

                        src = os.path.join(
                            alt_dir,
                            filename,
                        )

                        dst = os.path.join(
                            self.standards_dir,
                            filename,
                        )

                        if not os.path.exists(dst):

                            try:
                                shutil.copy2(
                                    src,
                                    dst,
                                )
                            except Exception:
                                pass

                pdf_files = [
                    os.path.join(
                        self.standards_dir,
                        filename,
                    )
                    for filename in os.listdir(
                        self.standards_dir
                    )
                    if filename.lower().endswith(".pdf")
                ]

        pdf_files.sort()

        print(
            f"[RAG] Standards directory: "
            f"{self.standards_dir}"
        )

        print(
            f"[RAG] Found {len(pdf_files)} PDF files."
        )

        self.chunks = []

        # =========================================================
        # 1. STRUCTURED STANDARDS KB
        # =========================================================

        standards = get_all_standards()

        for standard in standards:

            root_causes = standard.get(
                "root_cause_factors",
                [],
            )

            if not isinstance(root_causes, list):
                root_causes = [str(root_causes)]

            std_text = (
                f"Standard: "
                f"{standard.get('standard', '')}\n"

                f"Section: "
                f"{standard.get('section', '')}\n"

                f"Title: "
                f"{standard.get('title', '')}\n"

                f"Defect Type: "
                f"{standard.get('defect_type', '')}\n"

                f"Category: "
                f"{standard.get('category', '')}\n"

                f"Description: "
                f"{standard.get('description', '')}\n"

                f"Class 1 Criteria: "
                f"{standard.get('class_1_criteria', '')}\n"

                f"Class 2 Criteria: "
                f"{standard.get('class_2_criteria', '')}\n"

                f"Class 3 Criteria: "
                f"{standard.get('class_3_criteria', '')}\n"

                f"Acceptance Level: "
                f"{standard.get('acceptance_level', '')}\n"

                f"Severity: "
                f"{standard.get('severity_level', '')}\n"

                f"Electrical Impact: "
                f"{standard.get('electrical_impact', '')}\n"

                f"IPC-7711/7721 Rework: "
                f"{standard.get('ipc_7721_rework_procedure', '')}\n"

                f"Root Causes: "
                f"{'; '.join(root_causes)}"
            )

            self.chunks.append(
                {
                    "id": len(self.chunks),
                    "text": std_text,
                    "source": (
                        f"{standard.get('standard', '')} "
                        f"Section "
                        f"{standard.get('section', '')}"
                    ),
                    "source_file": (
                        "structured_standards_kb"
                    ),
                    "page": None,
                    "standard": standard,
                }
            )

        print(
            f"[RAG] Added "
            f"{len(self.chunks)} "
            f"structured standards clauses."
        )

        # =========================================================
        # 2. PDF DOCUMENTS
        # =========================================================

        for pdf_path in pdf_files:

            try:

                print(
                    f"[RAG] Processing: "
                    f"{os.path.basename(pdf_path)}"
                )

                pdf_chunks = (
                    self._extract_pdf_chunks(
                        pdf_path
                    )
                )

                self.chunks.extend(
                    pdf_chunks
                )

                print(
                    f"      chunks={len(pdf_chunks)}"
                )

            except Exception as exc:

                print(
                    f"[RAG] Failed to process "
                    f"{os.path.basename(pdf_path)}: "
                    f"{exc}"
                )

        # ---------------------------------------------------------
        # Re-number IDs globally.
        # ---------------------------------------------------------

        for index, chunk in enumerate(
            self.chunks
        ):
            chunk["id"] = index

        self.chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print(
            f"[RAG] Extracted total "
            f"{len(self.chunks)} chunks "
            f"(structured + PDF)."
        )

        if not self.chunks:

            print(
                "[RAG] WARNING: No usable chunks."
            )

            return

        self._build_faiss_index()

        self._build_bm25_index()

    # =============================================================
    # PDF EXTRACTION
    # =============================================================

    def _extract_pdf_chunks(
        self,
        pdf_path: str,
    ) -> List[Dict[str, Any]]:

        document = fitz.open(pdf_path)

        filename = os.path.basename(
            pdf_path
        )

        chunks = []

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                try:
                    text = page.get_text("text")
                except Exception:
                    text = ""

                if not text:
                    continue

                text = self._clean_text(text)

                if not text:
                    continue

                page_chunks = self._chunk_text(
                    text,
                    self.chunk_size,
                    self.chunk_overlap,
                )

                for chunk_index, chunk_text in enumerate(
                    page_chunks
                ):

                    chunks.append(
                        {
                            "id": len(chunks),
                            "source": filename,
                            "path": pdf_path,
                            "page": page_number,
                            "chunk_index": chunk_index,
                            "text": chunk_text,
                        }
                    )

        finally:

            document.close()

        return chunks

    # =============================================================
    # TEXT CLEANING
    # =============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        if not text:
            return ""

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        return text.strip()

    # =============================================================
    # CHUNKING
    # =============================================================

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:

        if not text:
            return []

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

        chunks = []

        start = 0

        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - overlap

        return chunks

    # =============================================================
    # FAISS
    # =============================================================

    def _build_faiss_index(self):

        if not self.chunk_texts:
            return

        print(
            "[RAG] Generating embeddings..."
        )

        embeddings = (
            self.embedding_model.encode(
                self.chunk_texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embedding matrix must be "
                "2-dimensional."
            )

        if (
            embeddings.shape[1]
            != self.embedding_dimension
        ):

            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected "
                f"{self.embedding_dimension}, "
                f"received "
                f"{embeddings.shape[1]}"
            )

        faiss.normalize_L2(
            embeddings
        )

        self.faiss_index = (
            faiss.IndexFlatIP(
                self.embedding_dimension
            )
        )

        self.faiss_index.add(
            embeddings
        )

        print(
            "[RAG] FAISS ready:"
        )

        print(
            f"      vectors="
            f"{self.faiss_index.ntotal}"
        )

        print(
            f"      dimension="
            f"{self.embedding_dimension}"
        )

    # =============================================================
    # BM25
    # =============================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:

        if not text:
            return []

        return re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            text.lower(),
        )

    def _build_bm25_index(self):

        if not self.chunk_texts:
            return

        tokenized_documents = [
            self._tokenize(text)
            for text in self.chunk_texts
        ]

        if RANK_BM25_AVAILABLE:

            self.bm25 = BM25Okapi(
                tokenized_documents
            )

            print(
                "[RAG] BM25 ready:"
            )

            print(
                f"      documents="
                f"{len(tokenized_documents)}"
            )

        else:

            self.bm25 = None

            print(
                "[RAG] WARNING: rank_bm25 "
                "is not installed."
            )

    # =============================================================
    # SEMANTIC SEARCH
    # =============================================================

    def _semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if (
            self.faiss_index is None
            or not self.chunks
            or not query
        ):
            return []

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:

            query_embedding = (
                query_embedding.reshape(
                    1,
                    -1,
                )
            )

        faiss.normalize_L2(
            query_embedding
        )

        k = min(
            max(top_k, 1),
            len(self.chunks),
        )

        scores, indices = (
            self.faiss_index.search(
                query_embedding,
                k,
            )
        )

        results = []

        for rank, (
            score,
            index,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):

            if index < 0:
                continue

            chunk = dict(
                self.chunks[index]
            )

            chunk["semantic_score"] = float(
                score
            )

            chunk["semantic_rank"] = rank

            results.append(chunk)

        return results

    # =============================================================
    # BM25 SEARCH
    # =============================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if (
            self.bm25 is None
            or not self.chunks
            or not query
        ):
            return []

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        k = min(
            max(top_k, 1),
            len(ranked_indices),
        )

        results = []

        rank = 0

        for index in ranked_indices[:k]:

            index = int(index)

            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            rank += 1

            chunk = dict(
                self.chunks[index]
            )

            chunk["bm25_score"] = score

            chunk["bm25_rank"] = rank

            results.append(chunk)

        return results

    # =============================================================
    # RRF
    # =============================================================

    def _rrf_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        fused = {}

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):

            chunk_id = int(
                result["id"]
            )

            if chunk_id not in fused:
                fused[chunk_id] = dict(
                    result
                )

            fused[chunk_id]["rrf_score"] = (
                fused[chunk_id].get(
                    "rrf_score",
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = int(
                result["id"]
            )

            if chunk_id not in fused:
                fused[chunk_id] = dict(
                    result
                )

            fused[chunk_id]["rrf_score"] = (
                fused[chunk_id].get(
                    "rrf_score",
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        results = sorted(
            fused.values(),
            key=lambda item: item.get(
                "rrf_score",
                0.0,
            ),
            reverse=True,
        )

        return results[:top_k]

    # =============================================================
    # DEFECT DETECTION
    # =============================================================

    @classmethod
    def _detect_defect(
        cls,
        query: str,
    ) -> Optional[str]:

        q = str(query or "").lower()

        # Most specific phrase first.
        all_aliases = []

        for defect, aliases in cls.DEFECT_ALIASES.items():

            for alias in aliases:

                all_aliases.append(
                    (
                        len(alias),
                        defect,
                        alias,
                    )
                )

        all_aliases.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, defect, alias in all_aliases:

            if alias in q:
                return defect

        return None

    # =============================================================
    # DEFECT ALIASES HELPER
    # =============================================================

    @classmethod
    def _aliases_for_defect(
        cls,
        defect: Optional[str],
    ) -> List[str]:

        if not defect:
            return []

        return cls.DEFECT_ALIASES.get(
            defect,
            [defect],
        )

    # =============================================================
    # QUERY INTENT
    # =============================================================

    @staticmethod
    def _detect_query_intent(
        query: str,
    ) -> str:

        q = str(query or "").strip().lower()

        # ---------------------------------------------------------
        # Definition
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "what is",
                "what are",
                "define",
                "definition of",
                "meaning of",
                "explain",
            ]
        ):
            return "definition"

        # ---------------------------------------------------------
        # Causes
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "what causes",
                "causes of",
                "why does",
                "why is",
                "reason for",
                "reasons for",
                "cause of",
            ]
        ):
            return "causes"

        # ---------------------------------------------------------
        # Rework
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "how to fix",
                "how do i fix",
                "how to repair",
                "how can i fix",
                "repair",
                "rework",
                "correct",
                "correction",
                "remove",
            ]
        ):
            return "rework"

        # ---------------------------------------------------------
        # Acceptance
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "acceptable",
                "acceptance",
                "acceptability",
                "can this pass",
                "should this pass",
                "pass or fail",
                "reject",
                "rejection",
                "compliant",
                "compliance",
            ]
        ):
            return "acceptance"

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "class 1",
                "class 2",
                "class 3",
                "product class",
                "classification",
            ]
        ):
            return "acceptance"

        # ---------------------------------------------------------
        # Inspection
        #
        # IMPORTANT:
        # "defect" alone is NOT an inspection intent.
        # A question such as "what is solder bridging?"
        # must remain a definition question.
        # ---------------------------------------------------------

        if any(
            phrase in q
            for phrase in [
                "inspection",
                "inspect",
                "inspection finding",
                "found during inspection",
                "detected during inspection",
                "how do i inspect",
                "how to inspect",
            ]
        ):
            return "inspection"

        return "general"

    # =============================================================
    # TEXT NORMALIZATION
    # =============================================================

    @staticmethod
    def _clean_text_for_answer(
        text: str,
    ) -> str:

        text = str(
            text or ""
        ).strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # =============================================================
    # STRUCTURED STANDARD LOOKUP
    # =============================================================

    @classmethod
    def _structured_standard_matches_defect(
        cls,
        standard: Optional[Dict[str, Any]],
        defect: Optional[str],
    ) -> bool:

        if not standard or not defect:
            return False

        fields = [
            standard.get(
                "defect_type",
                "",
            ),
            standard.get(
                "title",
                "",
            ),
            standard.get(
                "description",
                "",
            ),
        ]

        haystack = " ".join(
            str(value).lower()
            for value in fields
        )

        aliases = cls._aliases_for_defect(
            defect
        )

        return any(
            alias.lower() in haystack
            for alias in aliases
        )

    # =============================================================
    # DIRECT STRUCTURED STANDARD LOOKUP
    # =============================================================

    @classmethod
    def _get_structured_standard(
        cls,
        defect_type: str,
    ) -> Optional[Dict[str, Any]]:

        if not defect_type:
            return None

        normalized = (
            str(defect_type)
            .replace("_", " ")
            .strip()
            .lower()
        )

        # ---------------------------------------------------------
        # First: existing exact API
        # ---------------------------------------------------------

        try:

            standard = (
                get_standard_by_defect_type(
                    defect_type
                )
            )

            if standard is not None:

                if cls._structured_standard_matches_defect(
                    standard,
                    defect_type,
                ):
                    return standard

        except Exception:
            pass

        # ---------------------------------------------------------
        # Second: alias-aware scan.
        #
        # This is intentionally deterministic.
        # We do NOT use semantic search here.
        # ---------------------------------------------------------

        aliases = cls._aliases_for_defect(
            defect_type
        )

        for standard in get_all_standards():

            fields = [
                standard.get(
                    "defect_type",
                    "",
                ),
                standard.get(
                    "title",
                    "",
                ),
                standard.get(
                    "description",
                    "",
                ),
            ]

            haystack = " ".join(
                str(value).lower()
                for value in fields
            )

            if normalized in haystack:
                return standard

            if any(
                alias.lower() in haystack
                for alias in aliases
            ):
                return standard

        return None

    # =============================================================
    # SEARCH RERANKING
    # =============================================================

    def search_standards(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        top_k = max(
            int(top_k),
            1,
        )

        # ---------------------------------------------------------
        # Detect defect at retrieval level too.
        # ---------------------------------------------------------

        detected_defect = self._detect_defect(
            query
        )

        # ---------------------------------------------------------
        # Hybrid candidate retrieval
        # ---------------------------------------------------------

        candidate_k = max(
            top_k * 5,
            20,
        )

        semantic_results = (
            self._semantic_search(
                query,
                candidate_k,
            )
        )

        bm25_results = (
            self._bm25_search(
                query,
                candidate_k,
            )
        )

        fused_results = (
            self._rrf_fusion(
                semantic_results,
                bm25_results,
                candidate_k,
            )
        )

        if not fused_results:
            return []

        query_lower = query.lower()

        normalized_query = (
            query_lower
            .replace(
                "ipc a 610g",
                "ipc-a-610g",
            )
            .replace(
                "ipc a-610g",
                "ipc-a-610g",
            )
            .replace(
                "ipc a 610",
                "ipc-a-610",
            )
            .replace(
                "solder bridge",
                "solder bridging",
            )
        )

        query_terms = set(
            re.findall(
                r"\b[a-z0-9][a-z0-9_-]{2,}\b",
                normalized_query,
            )
        )

        technical_terms = {
            "acceptance",
            "acceptability",
            "criteria",
            "criterion",
            "acceptable",
            "unacceptable",
            "reject",
            "inspection",
            "defect",
            "defects",
            "solder",
            "bridging",
            "bridge",
            "short",
            "joint",
            "joints",
            "pad",
            "component",
            "assembly",
            "class",
            "severity",
            "electrical",
            "rework",
            "repair",
            "cause",
            "causes",
            "requirement",
            "requirements",
        }

        noise_phrases = [
            "standards checklist",
            "ipc reference standards",
            "typical process steps",
            "committee title",
            "meeting focus area",
            "table of contents",
            "contents",
            "reference standards",
            "list of standards",
            "standard title",
            "document title",
            "revision history",
            "bibliography",
        ]

        reranked = []

        for item in fused_results:

            raw_text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

            if not raw_text:
                continue

            text_lower = raw_text.lower()

            score = (
                float(
                    item.get(
                        "rrf_score",
                        0.0,
                    )
                )
                * 100.0
            )

            # -----------------------------------------------------
            # Query term overlap
            # -----------------------------------------------------

            text_terms = set(
                re.findall(
                    r"\b[a-z0-9][a-z0-9_-]{2,}\b",
                    text_lower,
                )
            )

            overlap = len(
                query_terms.intersection(
                    text_terms
                )
            )

            score += min(
                overlap * 1.5,
                18.0,
            )

            # -----------------------------------------------------
            # Technical content
            # -----------------------------------------------------

            for term in technical_terms:

                if term in text_lower:
                    score += 1.0

            if "acceptance criteria" in text_lower:
                score += 8.0

            if "acceptability" in text_lower:
                score += 7.0

            if "acceptable" in text_lower:
                score += 5.0

            if "unacceptable" in text_lower:
                score += 5.0

            if "shall" in text_lower:
                score += 2.0

            if "must" in text_lower:
                score += 2.0

            if "inspection" in text_lower:
                score += 3.0

            # -----------------------------------------------------
            # DEFECT-SPECIFIC BOOST
            # -----------------------------------------------------

            if detected_defect:

                aliases = (
                    self._aliases_for_defect(
                        detected_defect
                    )
                )

                exact_hits = sum(
                    1
                    for alias in aliases
                    if alias.lower()
                    in text_lower
                )

                if exact_hits > 0:

                    # Strong boost.
                    score += (
                        18.0
                        * exact_hits
                    )

                else:

                    # Do not allow unrelated generic PCB chunks
                    # to outrank a chunk about the requested defect.
                    score -= 8.0

                # Strong penalty for clearly different defects.
                for other_defect in (
                    self.UNRELATED_DEFECTS
                ):

                    if other_defect in aliases:
                        continue

                    if other_defect in text_lower:

                        score -= 5.0

            # -----------------------------------------------------
            # Acceptance language
            # -----------------------------------------------------

            if (
                "acceptance"
                in query_lower
                or "acceptable"
                in query_lower
                or "pass"
                in query_lower
            ):

                if "acceptance criteria" in text_lower:
                    score += 10.0

                if "acceptable" in text_lower:
                    score += 7.0

                if "unacceptable" in text_lower:
                    score += 7.0

            # -----------------------------------------------------
            # IPC relevance
            # -----------------------------------------------------

            if "ipc-a-610g" in query_lower:

                if "ipc-a-610g" in text_lower:
                    score += 30.0

                elif "ipc-a-610" in text_lower:
                    score += 5.0

                if "ipc-a-610c" in text_lower:
                    score -= 8.0

            elif "ipc-a-610" in query_lower:

                if "ipc-a-610" in text_lower:
                    score += 15.0

            # -----------------------------------------------------
            # Class relevance
            # -----------------------------------------------------

            class_match = re.search(
                r"class\s*([123])",
                query_lower,
            )

            if class_match:

                requested_class = (
                    class_match.group(1)
                )

                if re.search(
                    rf"class\s*{requested_class}\b",
                    text_lower,
                ):
                    score += 8.0

            # -----------------------------------------------------
            # Penalize navigation/reference material
            # -----------------------------------------------------

            for phrase in noise_phrases:

                if phrase in text_lower:
                    score -= 12.0

            standard_id_count = len(
                re.findall(
                    r"\b(?:ipc|j-std|ansi|iso)-?[a-z0-9-]+\b",
                    text_lower,
                )
            )

            if standard_id_count >= 6:
                score -= 8.0

            if standard_id_count >= 10:
                score -= 10.0

            # -----------------------------------------------------
            # Prefer substantive prose
            # -----------------------------------------------------

            sentence_count = len(
                re.findall(
                    r"[.!?]",
                    raw_text,
                )
            )

            if sentence_count >= 2:
                score += 2.0

            if len(raw_text) < 120:
                score -= 3.0

            item = dict(item)

            item["retrieval_score"] = float(
                score
            )

            reranked.append(item)

        reranked.sort(
            key=lambda item: item.get(
                "retrieval_score",
                0.0,
            ),
            reverse=True,
        )

        return reranked[:top_k]

    # =============================================================
    # STRICT EVIDENCE FILTER
    # =============================================================

    @classmethod
    def _find_relevant_evidence(
        cls,
        evidence: List[Dict[str, Any]],
        defect: Optional[str],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:

        if not evidence:
            return []

        if not defect:
            return evidence[:limit]

        aliases = cls._aliases_for_defect(
            defect
        )

        ranked = []

        for item in evidence:

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).lower()

            exact_hits = sum(
                1
                for alias in aliases
                if alias.lower() in text
            )

            # -----------------------------------------------------
            # CRITICAL FIX:
            #
            # If the user explicitly asked about a defect,
            # unrelated chunks are NOT valid evidence.
            # -----------------------------------------------------

            if exact_hits == 0:
                continue

            score = float(
                item.get(
                    "retrieval_score",
                    0.0,
                )
            )

            score += (
                exact_hits * 35.0
            )

            # Penalize different defect terminology.
            for other_defect in (
                cls.UNRELATED_DEFECTS
            ):

                if other_defect in aliases:
                    continue

                if other_defect in text:
                    score -= 8.0

            ranked.append(
                (
                    score,
                    exact_hits,
                    item,
                )
            )

        ranked.sort(
            key=lambda value: (
                value[0],
                value[1],
            ),
            reverse=True,
        )

        return [
            item
            for _, _, item in ranked[:limit]
        ]

    # =============================================================
    # AUDIT EVIDENCE
    # =============================================================

    def retrieve_audit_evidence(
        self,
        defect_type: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not defect_type:
            return []

        clean_defect = (
            str(defect_type)
            .replace("_", " ")
            .strip()
        )

        query = (
            f"PCB manufacturing "
            f"{clean_defect} "
            f"assembly quality requirements "
            f"acceptance criteria "
            f"defect severity "
            f"electrical impact "
            f"corrective action"
        )

        evidence = self.search_standards(
            query,
            top_k,
        )

        detected = self._detect_defect(
            clean_defect
        )

        if detected:

            evidence = (
                self._find_relevant_evidence(
                    evidence,
                    detected,
                    limit=top_k,
                )
            )

        return evidence

    # =============================================================
    # EVIDENCE SUMMARY
    # =============================================================

    @staticmethod
    def _build_evidence_summary(
        evidence: List[Dict[str, Any]],
    ) -> str:

        if not evidence:

            return (
                "No matching standards evidence "
                "was retrieved."
            )

        lines = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            source = item.get(
                "source",
                "Unknown source",
            )

            page = item.get(
                "page",
                "?",
            )

            text = item.get(
                "text",
                "",
            ).strip()

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            if len(text) > 500:

                text = (
                    text[:500].rstrip()
                    + "..."
                )

            lines.append(
                f"{index}. "
                f"{source} "
                f"(page {page}): "
                f"{text}"
            )

        return "\n".join(lines)

    # =============================================================
    # AUDIT FINDING
    # =============================================================

    def synthesize_audit_finding(
        self,
        defect_type: str,
        predicted_class: str = "Anomaly",
        confidence: float = 0.0,
        visual_findings: Optional[
            Dict[str, Any]
        ] = None,
        operating_class: str = "Class 3",
        top_k: int = 5,
    ) -> Dict[str, Any]:

        visual_findings = (
            visual_findings
            if isinstance(
                visual_findings,
                dict,
            )
            else {}
        )

        clean_defect = (
            str(defect_type)
            .replace("_", " ")
            .strip()
        )

        detected_defect = (
            self._detect_defect(
                clean_defect
            )
            or clean_defect
        )

        standard = (
            self._get_structured_standard(
                detected_defect
            )
        )

        defect_area_ratio = (
            visual_findings.get(
                "defect_area_ratio",
                0.0,
            )
        )

        try:

            defect_area_ratio = float(
                defect_area_ratio
            )

        except (
            TypeError,
            ValueError,
        ):

            defect_area_ratio = 0.0

        defect_area_ratio = max(
            0.0,
            min(
                defect_area_ratio,
                1.0,
            ),
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        rag_query = (
            f"PCB manufacturing "
            f"{clean_defect}. "
            f"Assembly quality requirements, "
            f"acceptance criteria, "
            f"defect severity, "
            f"electrical impact and corrective action. "
            f"Operating class: "
            f"{operating_class}. "
            f"Detected confidence: "
            f"{confidence:.3f}. "
            f"Estimated visual defect area ratio: "
            f"{defect_area_ratio:.4f}."
        )

        evidence = self.search_standards(
            rag_query,
            max(top_k, 5),
        )

        # Strict defect filter.
        detected_for_filter = (
            self._detect_defect(
                clean_defect
            )
        )

        if detected_for_filter:

            evidence = (
                self._find_relevant_evidence(
                    evidence,
                    detected_for_filter,
                    limit=top_k,
                )
            )

        evidence_summary = (
            self._build_evidence_summary(
                evidence
            )
        )

        has_pdf_evidence = any(
            item.get("source_file")
            != "structured_standards_kb"
            for item in evidence
        )

        has_structured_evidence = any(
            item.get("source_file")
            == "structured_standards_kb"
            for item in evidence
        )

        if has_pdf_evidence:

            knowledge_source = (
                "Manufacturer / technical PDF "
                "standards via hybrid "
                "embedding + FAISS + BM25 retrieval"
            )

        elif standard:

            knowledge_source = (
                "Structured PCB standards "
                "knowledge base"
            )

        else:

            knowledge_source = (
                "No standards evidence available"
            )

        result = {

            "defect_type":
                defect_type,

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "operating_class":
                operating_class,

            "visual_findings":
                visual_findings,

            "rag_query":
                rag_query,

            "knowledge_source":
                knowledge_source,

            "retrieved_evidence":
                evidence,

            "evidence_summary":
                evidence_summary,

            "evidence_count":
                len(evidence),

            "grounding": {

                "pdf_evidence_used":
                    has_pdf_evidence,

                "structured_kb_used":
                    bool(
                        standard
                        or has_structured_evidence
                    ),

                "llm_generated":
                    False,
            },
        }

        # ---------------------------------------------------------
        # Structured standard information
        # ---------------------------------------------------------

        if standard:

            severity = standard.get(
                "severity_level",
                standard.get(
                    "severity",
                    standard.get(
                        "category",
                        "Unknown",
                    ),
                ),
            )

            rework_procedure = (
                standard.get(
                    "ipc_7721_rework_procedure",
                    standard.get(
                        "rework_procedure",
                        "",
                    ),
                )
            )

            result["standard"] = {

                "id":
                    standard.get(
                        "id",
                        "",
                    ),

                "standard":
                    standard.get(
                        "standard",
                        "",
                    ),

                "section":
                    standard.get(
                        "section",
                        "",
                    ),

                "title":
                    standard.get(
                        "title",
                        "",
                    ),

                "category":
                    standard.get(
                        "category",
                        "",
                    ),

                "defect_type":
                    standard.get(
                        "defect_type",
                        "",
                    ),

                "description":
                    standard.get(
                        "description",
                        "",
                    ),

                "severity_level":
                    severity,

                "acceptance_level":
                    standard.get(
                        "acceptance_level",
                        "",
                    ),

                "risk_factor":
                    standard.get(
                        "risk_factor",
                        0.0,
                    ),

                "electrical_impact":
                    standard.get(
                        "electrical_impact",
                        "",
                    ),

                "rework_procedure":
                    rework_procedure,

                "root_cause_factors":
                    standard.get(
                        "root_cause_factors",
                        [],
                    ),
            }

        else:

            result["standard"] = None

        # ---------------------------------------------------------
        # Finding
        # ---------------------------------------------------------

        if str(
            predicted_class
        ).lower() in {
            "normal",
            "good_assembly",
            "good assembly",
        }:

            result["finding"] = (
                "The visual inspection model "
                "classified the PCB region as "
                "acceptable / normal."
            )

        else:

            severity_text = "anomaly"

            if standard:

                severity_text = str(
                    standard.get(
                        "severity_level",
                        standard.get(
                            "severity",
                            standard.get(
                                "category",
                                "anomaly",
                            ),
                        ),
                    )
                )

            result["finding"] = (
                f"The visual inspection model "
                f"detected {clean_defect} "
                f"with confidence "
                f"{confidence:.3f}. "
                f"The structured standards "
                f"knowledge base categorizes "
                f"this finding as "
                f"{severity_text}. "
                f"Applicable manufacturing "
                f"evidence was retrieved from "
                f"the standards corpus."
            )

        return result

    # =============================================================
    # USER STANDARDS QUERY
    # =============================================================

    def answer_standards_query(
        self,
        query: str,
        top_k: int = 5,
        operating_class: str = "Class 3",
    ) -> Dict[str, Any]:

        if not query or not query.strip():

            return {
                "query": "",
                "answer": (
                    "Please provide a PCB manufacturing "
                    "or standards question."
                ),
                "retrieved_evidence": [],
                "evidence_count": 0,
                "operating_class": operating_class,
                "intent": "general",
                "detected_defect": None,
                "grounding": {
                    "pdf_evidence_used": False,
                    "structured_kb_used": False,
                    "llm_generated": False,
                },
            }

        query_clean = query.strip()

        # ---------------------------------------------------------
        # 1. Understand question.
        # ---------------------------------------------------------

        # Detect the defect using the existing RAG detector.
        defect = (
            self._detect_defect(
                query_clean
            )
        )

        # ---------------------------------------------------------
        # Acceptance/rejection questions must take priority over
        # generic "why" / causes interpretation.
        #
        # This only controls retrieval intent. The final answer
        # still comes entirely from retrieved standards evidence.
        # ---------------------------------------------------------

        _q_lower = query_clean.lower()

        if any(term in _q_lower for term in [
            "unacceptable",
            "not acceptable",
            "acceptable",
            "acceptance",
            "acceptability",
            "can this pass",
            "should this pass",
            "pass or fail",
            "reject",
            "rejection",
            "non-conformance",
            "nonconformance",
            "non-compliance",
            "noncompliance",
            "compliant",
            "compliance",
            "violation",
            "fails acceptance"
        ]):
            intent = "acceptance"

        else:
            # Preserve the existing RAG intent detector for all
            # other questions.
            intent = (
                self._detect_query_intent(
                    query_clean
                )
            )

        # ---------------------------------------------------------
        # 2. Build focused retrieval query.
        # ---------------------------------------------------------

        retrieval_query = query_clean

        if defect:

            retrieval_query = (
                f"{query_clean}. "
                f"PCB defect: {defect}."
            )

        if intent == "definition":

            retrieval_query += (
                f" Definition and description "
                f"of {defect or 'the PCB condition'}."
            )

        elif intent == "causes":

            retrieval_query += (
                f" Manufacturing causes and prevention "
                f"of {defect or 'this defect'}."
            )

        elif intent == "rework":

            retrieval_query += (
                f" Corrective action and rework procedure "
                f"for {defect or 'this defect'}."
            )

        elif intent == "acceptance":

            retrieval_query += (
                f" Acceptance criteria for "
                f"{operating_class}."
            )

        elif intent == "inspection":

            retrieval_query += (
                f" Inspection requirements and assessment "
                f"for {defect or 'this condition'}."
            )

        # ---------------------------------------------------------
        # 3. Hybrid retrieval.
        # ---------------------------------------------------------

        evidence = self.search_standards(
            retrieval_query,
            max(
                int(top_k),
                5,
            ),
        )

        # ---------------------------------------------------------
        # 4. STRICT defect evidence filtering.
        #
        # This is the major fix.
        #
        # Example:
        #
        # "What is solder bridging?"
        #
        # A side-overhang chunk that happens to contain
        # "assembly", "solder", "acceptance", etc. is discarded.
        # ---------------------------------------------------------

        if defect:

            evidence = (
                self._find_relevant_evidence(
                    evidence,
                    defect,
                    limit=max(
                        int(top_k),
                        5,
                    ),
                )
            )

        # ---------------------------------------------------------
        # 5. Direct structured KB lookup.
        #
        # IMPORTANT:
        # Do NOT depend on semantic KB search for a known defect.
        # ---------------------------------------------------------

        matched_std = None

        if defect:

            candidate = (
                self._get_structured_standard(
                    defect
                )
            )

            if (
                candidate
                and self._structured_standard_matches_defect(
                    candidate,
                    defect,
                )
            ):

                matched_std = candidate

        else:

            # Only use generic semantic KB lookup when there is
            # no explicit defect in the user's question.
            try:

                kb_matches = search_standards_kb(
                    query_clean,
                    top_k=5,
                )

            except Exception:

                kb_matches = []

            if kb_matches:
                matched_std = kb_matches[0]

        # ---------------------------------------------------------
        # 6. Prefer structured standard from exact defect.
        # ---------------------------------------------------------

        if defect:

            for item in evidence:

                candidate = item.get(
                    "standard"
                )

                if (
                    candidate
                    and self._structured_standard_matches_defect(
                        candidate,
                        defect,
                    )
                ):

                    matched_std = candidate
                    break

        # ---------------------------------------------------------
        # 7. Build answer.
        # ---------------------------------------------------------

        answer_parts = []

        display_name = (
            defect.title()
            if defect
            else "PCB Manufacturing"
        )

        # =========================================================
        # DEFINITION
        # =========================================================

        if intent == "definition":

            if matched_std:

                title = matched_std.get(
                    "title",
                    display_name,
                )

                description = (
                    matched_std.get(
                        "description",
                        "",
                    )
                )

                standard_name = (
                    matched_std.get(
                        "standard",
                        "",
                    )
                )

                section = (
                    matched_std.get(
                        "section",
                        "",
                    )
                )

                answer_parts.append(
                    f"### {title}"
                )

                if description:

                    answer_parts.append(
                        self._clean_text_for_answer(
                            description
                        )
                    )

                if standard_name:

                    reference = (
                        f"**Standard reference:** "
                        f"{standard_name}"
                    )

                    if section:
                        reference += (
                            f" Section {section}"
                        )

                    answer_parts.append(
                        reference
                    )

            elif evidence:

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 500:

                    text = (
                        text[:500]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(
                    f"### {display_name}"
                )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    f"### {display_name}"
                )

                answer_parts.append(
                    "I couldn't find a sufficiently "
                    "specific standards reference for "
                    "this term in the available corpus."
                )

        # =========================================================
        # CAUSES
        # =========================================================

        elif intent == "causes":

            answer_parts.append(
                f"### Common Causes of "
                f"{display_name}"
            )

            causes = []

            if matched_std:

                causes = matched_std.get(
                    "root_cause_factors",
                    [],
                )

            if causes:

                for cause in causes[:6]:

                    answer_parts.append(
                        f"- {cause}"
                    )

            elif evidence:

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 700:

                    text = (
                        text[:700]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    "The available standards corpus "
                    "does not provide a sufficiently "
                    "specific cause list for this condition."
                )

        # =========================================================
        # REWORK
        # =========================================================

        elif intent == "rework":

            answer_parts.append(
                "### Corrective Action / Rework"
            )

            rework = ""

            if matched_std:

                rework = matched_std.get(
                    "ipc_7721_rework_procedure",
                    matched_std.get(
                        "rework_procedure",
                        "",
                    ),
                )

            if rework:

                answer_parts.append(
                    self._clean_text_for_answer(
                        rework
                    )
                )

            elif evidence:

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 700:

                    text = (
                        text[:700]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    "No specific rework procedure "
                    "was found in the available "
                    "standards evidence."
                )

        # =========================================================
        # ACCEPTANCE
        # =========================================================

        elif intent == "acceptance":

            answer_parts.append(
                "### Acceptance Criteria"
            )

            if matched_std:

                c1 = matched_std.get(
                    "class_1_criteria",
                    "",
                )

                c2 = matched_std.get(
                    "class_2_criteria",
                    "",
                )

                c3 = matched_std.get(
                    "class_3_criteria",
                    "",
                )

                requested_class = (
                    operating_class.lower()
                )

                if "class 1" in requested_class:
                    criterion = c1

                elif "class 2" in requested_class:
                    criterion = c2

                else:
                    criterion = c3

                if criterion:

                    answer_parts.append(
                        f"**{operating_class}:** "
                        f"{criterion}"
                    )

                else:

                    answer_parts.append(
                        "The available structured standard "
                        "does not contain a specific criterion "
                        f"for {operating_class}."
                    )

                standard_name = matched_std.get(
                    "standard",
                    "",
                )

                section = matched_std.get(
                    "section",
                    "",
                )

                if standard_name:

                    reference = (
                        f"**Reference:** "
                        f"{standard_name}"
                    )

                    if section:

                        reference += (
                            f" Section {section}"
                        )

                    answer_parts.append(
                        reference
                    )

            elif evidence:

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 800:

                    text = (
                        text[:800]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    "I could not locate a sufficiently "
                    "specific acceptance criterion for "
                    "this condition."
                )

        # =========================================================
        # INSPECTION
        # =========================================================

        elif intent == "inspection":

            answer_parts.append(
                "### Inspection Guidance"
            )

            if matched_std:

                description = matched_std.get(
                    "description",
                    "",
                )

                if description:

                    answer_parts.append(
                        self._clean_text_for_answer(
                            description
                        )
                    )

                criterion_key = {
                    "Class 1":
                        "class_1_criteria",
                    "Class 2":
                        "class_2_criteria",
                    "Class 3":
                        "class_3_criteria",
                }.get(
                    operating_class,
                    "class_3_criteria",
                )

                criterion = matched_std.get(
                    criterion_key,
                    "",
                )

                if criterion:

                    answer_parts.append(
                        f"**{operating_class} "
                        f"criterion:** "
                        f"{criterion}"
                    )

            elif evidence:

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 800:

                    text = (
                        text[:800]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    "No sufficiently specific inspection "
                    "evidence was found."
                )

        # =========================================================
        # GENERAL
        # =========================================================

        else:

            if matched_std:

                title = matched_std.get(
                    "title",
                    display_name,
                )

                description = matched_std.get(
                    "description",
                    "",
                )

                answer_parts.append(
                    f"### {title}"
                )

                if description:

                    answer_parts.append(
                        self._clean_text_for_answer(
                            description
                        )
                    )

                if evidence:

                    text = (
                        self._clean_text_for_answer(
                            evidence[0].get(
                                "text",
                                "",
                            )
                        )
                    )

                    if len(text) > 500:

                        text = (
                            text[:500]
                            .rsplit(
                                " ",
                                1,
                            )[0]
                            + "..."
                        )

                    answer_parts.append(
                        f"**Supporting evidence:** "
                        f"{text}"
                    )

            elif evidence:

                answer_parts.append(
                    "### Standards Evidence"
                )

                text = (
                    self._clean_text_for_answer(
                        evidence[0].get(
                            "text",
                            "",
                        )
                    )
                )

                if len(text) > 800:

                    text = (
                        text[:800]
                        .rsplit(
                            " ",
                            1,
                        )[0]
                        + "..."
                    )

                answer_parts.append(text)

            else:

                answer_parts.append(
                    "I couldn't find a sufficiently "
                    "specific standard reference for "
                    "that question in the available "
                    "standards corpus."
                )

        # ---------------------------------------------------------
        # Helpful follow-up.
        # ---------------------------------------------------------

        if intent == "definition" and defect:

            answer_parts.append(
                "\nIf you want, I can also check the "
                f"**{operating_class} acceptance criteria** "
                f"for {defect}."
            )

        elif intent == "causes" and defect:

            answer_parts.append(
                "\nIf you're investigating an actual "
                "PCB finding, I can also check its "
                "acceptance criteria."
            )

        # ---------------------------------------------------------
        # Final answer
        # ---------------------------------------------------------

        answer = "\n\n".join(
            part.strip()
            for part in answer_parts
            if part and part.strip()
        )

        # ---------------------------------------------------------
        # Evidence metadata
        # ---------------------------------------------------------

        formatted_evidence = []

        for item in evidence[:5]:

            text = (
                self._clean_text_for_answer(
                    item.get(
                        "text",
                        "",
                    )
                )
            )

            if len(text) > 250:

                text = (
                    text[:250]
                    .rsplit(
                        " ",
                        1,
                    )[0]
                    + "..."
                )

            formatted_evidence.append(
                {
                    "source": item.get(
                        "source",
                        "Standards Library",
                    ),
                    "source_file": item.get(
                        "source_file",
                        "Technical Guideline",
                    ),
                    "page": item.get(
                        "page"
                    ),
                    "text": text,
                    "retrieval_score": item.get(
                        "retrieval_score"
                    ),
                }
            )

        pdf_evidence_used = any(
            item.get(
                "source_file"
            ) != "structured_standards_kb"
            for item in evidence
        )

        structured_kb_used = (
            matched_std is not None
            or any(
                item.get(
                    "source_file"
                ) == "structured_standards_kb"
                for item in evidence
            )
        )

        return {

            "query":
                query_clean,

            "answer":
                answer,

            "retrieved_evidence":
                formatted_evidence,

            "evidence_count":
                len(formatted_evidence),

            "operating_class":
                operating_class,

            "intent":
                intent,

            "detected_defect":
                defect,

            "grounding": {

                "pdf_evidence_used":
                    pdf_evidence_used,

                "structured_kb_used":
                    structured_kb_used,

                "llm_generated":
                    False,
            },
        }

    # =============================================================
    # PDF SOURCES
    # =============================================================

    def list_pdf_sources(
        self,
    ) -> List[str]:

        if not os.path.isdir(
            self.standards_dir
        ):
            return []

        return sorted(
            [
                filename
                for filename in os.listdir(
                    self.standards_dir
                )
                if filename.lower().endswith(
                    ".pdf"
                )
            ]
        )

    # =============================================================
    # SYSTEM STATUS
    # =============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:

        return {

            "standards_directory":
                self.standards_dir,

            "embedding_model":
                self.embedding_model_name,

            "embedding_dimension":
                self.embedding_dimension,

            "cache_directory":
                os.environ.get(
                    "HF_HOME"
                ),

            "pdf_sources":
                self.list_pdf_sources(),

            "pdf_count":
                len(
                    self.list_pdf_sources()
                ),

            "pdf_chunk_count":
                len(
                    self.chunks
                ),

            "faiss_ready":
                self.faiss_index is not None,

            "faiss_vectors":
                (
                    int(
                        self.faiss_index.ntotal
                    )
                    if self.faiss_index is not None
                    else 0
                ),

            "bm25_ready":
                self.bm25 is not None,

            "rrf_k":
                self.rrf_k,

            "chunk_size":
                self.chunk_size,

            "chunk_overlap":
                self.chunk_overlap,
        }


# =============================================================
# LOCAL TEST
# =============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "ManufacturingRAG-QA — Standards Assistant Test"
    )

    print(
        "=" * 70
    )

    engine = StandardsRAGEngine()

    print(
        "\n[RAG STATUS]"
    )

    print(
        engine.get_status()
    )

    # ---------------------------------------------------------
    # Conversational tests
    # ---------------------------------------------------------

    test_queries = [

        "What is solder bridging?",

        "What causes solder balls?",

        "Is solder bridging acceptable for Class 3?",

        "How do I fix solder bridging?",

        "What is tombstoning?",

        "What causes poor wetting?",

    ]

    for test_query in test_queries:

        print(
            "\n" + "=" * 70
        )

        print(
            "[TEST QUERY]"
        )

        print(
            test_query
        )

        result = (
            engine.answer_standards_query(
                test_query,
                top_k=3,
                operating_class="Class 3",
            )
        )

        print(
            "\n[INTENT]"
        )

        print(
            result.get(
                "intent"
            )
        )

        print(
            "\n[DETECTED DEFECT]"
        )

        print(
            result.get(
                "detected_defect"
            )
        )

        print(
            "\n[ANSWER]"
        )

        print(
            result.get(
                "answer"
            )
        )

        print(
            "\n[EVIDENCE COUNT]"
        )

        print(
            result.get(
                "evidence_count"
            )
        )

        print(
            "\n[GROUNDING]"
        )

        print(
            result.get(
                "grounding"
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "Standards Assistant tests completed."
    )

    print(
        "=" * 70
    )