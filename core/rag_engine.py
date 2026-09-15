"""
Standards-grounded RAG engine for ManufacturingRAG-QA.

Pipeline:
    PDF standards
        â†“
    Page-aware text extraction
        â†“
    Chunking
        â†“
    Embeddings
        â†“
    FAISS semantic retrieval
        +
    BM25 lexical retrieval
        â†“
    Reciprocal Rank Fusion (RRF)
        â†“
    Standards-grounded audit evidence

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
    get_standard_by_id,
    search_standards_kb,
)


class StandardsRAGEngine:
    """
    PDF-grounded hybrid Retrieval-Augmented Generation engine.

    Uses:
        - configurable lightweight embedding model
        - FAISS for semantic retrieval
        - BM25 for lexical retrieval
        - Reciprocal Rank Fusion for hybrid ranking
    """

    def __init__(
        self,
        standards_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        rrf_k: int = 60,
    ):

        # ---------------------------------------------------------
        # Paths
        # ---------------------------------------------------------

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                ".."
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
        #
        # Priority:
        #
        #   explicit constructor argument
        #       â†“
        #   RAG_EMBEDDING_MODEL
        #       â†“
        #   lightweight MiniLM default
        #
        # Do NOT default to BGE-M3 because it is very large.
        # ---------------------------------------------------------

        if embedding_model is None:

            embedding_model = os.environ.get(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        self.embedding_model_name = (
            embedding_model
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------

        self.embedding_model = None

        self.chunks: List[
            Dict[str, Any]
        ] = []

        self.chunk_texts: List[str] = []

        self.faiss_index = None

        self.bm25 = None

        self.embedding_dimension: Optional[
            int
        ] = None

        # ---------------------------------------------------------
        # Initialize
        # ---------------------------------------------------------

        self._load_embedding_model()

        self._build_knowledge_base()

    # =============================================================
    # EMBEDDING MODEL
    # =============================================================

    def _load_embedding_model(self):

        print(
            "[RAG] Loading embedding model:"
        )

        print(
            f"      {self.embedding_model_name}"
        )

        # ---------------------------------------------------------
        # Display cache location
        # ---------------------------------------------------------

        hf_home = os.environ.get(
            "HF_HOME"
        )

        if hf_home:

            print(
                "[RAG] Hugging Face cache:"
            )

            print(
                f"      {hf_home}"
            )

        # ---------------------------------------------------------
        # Load model (prefer local snapshot to bypass SSL issues on Windows)
        # ---------------------------------------------------------
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_path = self.embedding_model_name

        if hf_home:
            # Check for local cached snapshot
            model_folder = "models--" + self.embedding_model_name.replace("/", "--")
            snapshots_dir = os.path.join(hf_home, "hub", model_folder, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = [
                    os.path.join(snapshots_dir, s)
                    for s in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, s))
                ]
                if snapshots:
                    model_path = snapshots[0]
                    print(f"[RAG] Using local snapshot: {model_path}")

        try:
            self.embedding_model = SentenceTransformer(model_path)
        except Exception:
            # Fallback
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # ---------------------------------------------------------
        # Test embedding
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
                    -1
                )
            )

        self.embedding_dimension = (
            int(
                test_embedding.shape[1]
            )
        )

        print(
            "[RAG] Embedding model ready:"
        )

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
            exist_ok=True
        )

        pdf_files = [
            os.path.join(
                self.standards_dir,
                filename
            )
            for filename in os.listdir(
                self.standards_dir
            )
            if filename.lower().endswith(".pdf")
        ]

        # Auto-fallback: if local directory has no PDFs, check alternate directory
        if not pdf_files:
            alt_dir = "D:/ManufacturingRAG-QA/data/standards"
            if os.path.exists(alt_dir):
                import shutil
                for fn in os.listdir(alt_dir):
                    if fn.lower().endswith(".pdf"):
                        src_f = os.path.join(alt_dir, fn)
                        dst_f = os.path.join(self.standards_dir, fn)
                        if not os.path.exists(dst_f):
                            try:
                                shutil.copy2(src_f, dst_f)
                            except Exception:
                                pass
                pdf_files = [
                    os.path.join(self.standards_dir, fn)
                    for fn in os.listdir(self.standards_dir)
                    if fn.lower().endswith(".pdf")
                ]

        pdf_files.sort()

        print(
            f"[RAG] Standards directory: {self.standards_dir}"
        )
        print(
            f"[RAG] Found {len(pdf_files)} PDF files."
        )

        self.chunks = []

        # ---------------------------------------------------------
        # 1. Index structured standards knowledge base entries
        # ---------------------------------------------------------
        for std in get_all_standards():
            std_text = (
                f"Standard: {std.get('standard')} Section: {std.get('section')}\n"
                f"Title: {std.get('title')} ({std.get('defect_type', '')})\n"
                f"Category: {std.get('category')}\n"
                f"Description: {std.get('description')}\n"
                f"Class 1 Criteria: {std.get('class_1_criteria')}\n"
                f"Class 2 Criteria: {std.get('class_2_criteria')}\n"
                f"Class 3 Criteria: {std.get('class_3_criteria')}\n"
                f"Acceptance Level: {std.get('acceptance_level')}\n"
                f"Severity: {std.get('severity_level')}\n"
                f"Electrical Impact: {std.get('electrical_impact')}\n"
                f"IPC-7711/7721 Rework: {std.get('ipc_7721_rework_procedure')}\n"
                f"Root Causes: {'; '.join(std.get('root_cause_factors', []))}"
            )
            self.chunks.append({
                "id": len(self.chunks),
                "text": std_text,
                "source": f"{std.get('standard')} Section {std.get('section')}",
                "source_file": "structured_standards_kb",
                "page": None,
                "standard": std
            })

        print(
            f"[RAG] Added {len(self.chunks)} structured standards clauses."
        )

        # ---------------------------------------------------------
        # 2. Index PDF documents
        # ---------------------------------------------------------
        for pdf_path in pdf_files:
            try:
                print(
                    f"[RAG] Processing: {os.path.basename(pdf_path)}"
                )
                pdf_chunks = self._extract_pdf_chunks(pdf_path)
                self.chunks.extend(pdf_chunks)
                print(f"      chunks={len(pdf_chunks)}")
            except Exception as exc:
                print(
                    f"[RAG] Failed to process {os.path.basename(pdf_path)}: {exc}"
                )

        # ---------------------------------------------------------
        # Re-number chunk IDs globally.
        # ---------------------------------------------------------
        for index, chunk in enumerate(self.chunks):
            chunk["id"] = index

        self.chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print(
            f"[RAG] Extracted total {len(self.chunks)} chunks (structured + PDF)."
        )

        if not self.chunks:
            print("[RAG] WARNING: No usable chunks.")
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

        document = fitz.open(
            pdf_path
        )

        filename = os.path.basename(
            pdf_path
        )

        chunks = []

        try:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                try:

                    text = page.get_text(
                        "text"
                    )

                except Exception:

                    text = ""

                if not text:

                    continue

                text = self._clean_text(
                    text
                )

                if not text:

                    continue

                page_chunks = (
                    self._chunk_text(
                        text,
                        self.chunk_size,
                        self.chunk_overlap,
                    )
                )

                for chunk_index, chunk_text in enumerate(
                    page_chunks
                ):

                    chunks.append(
                        {
                            "id":
                                len(chunks),

                            "source":
                                filename,

                            "path":
                                pdf_path,

                            "page":
                                page_number,

                            "chunk_index":
                                chunk_index,

                            "text":
                                chunk_text,
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
        text: str
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
                "chunk_overlap must be smaller than chunk_size."
            )

        chunks = []

        start = 0

        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length
            )

            chunk = (
                text[start:end]
                .strip()
            )

            if chunk:

                chunks.append(
                    chunk
                )

            if end >= text_length:

                break

            start = (
                end - overlap
            )

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
        text: str
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
    # SEMANTIC RETRIEVAL
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
                    -1
                )
            )

        faiss.normalize_L2(
            query_embedding
        )

        k = min(
            max(top_k, 1),
            len(self.chunks)
        )

        scores, indices = (
            self.faiss_index.search(
                query_embedding,
                k
            )
        )

        results = []

        for rank, (
            score,
            index
        ) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            if index < 0:

                continue

            chunk = dict(
                self.chunks[index]
            )

            chunk["semantic_score"] = (
                float(score)
            )

            chunk["semantic_rank"] = (
                rank
            )

            results.append(
                chunk
            )

        return results

    # =============================================================
    # BM25 RETRIEVAL
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

        query_tokens = (
            self._tokenize(query)
        )

        if not query_tokens:

            return []

        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )

        ranked_indices = (
            np.argsort(scores)[::-1]
        )

        k = min(
            max(top_k, 1),
            len(ranked_indices)
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

            chunk["bm25_score"] = (
                score
            )

            chunk["bm25_rank"] = (
                rank
            )

            results.append(
                chunk
            )

        return results

    # =============================================================
    # RRF
    # =============================================================

    def _rrf_fusion(
        self,
        semantic_results: List[
            Dict[str, Any]
        ],
        bm25_results: List[
            Dict[str, Any]
        ],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        fused = {}

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            chunk_id = int(
                result["id"]
            )

            if chunk_id not in fused:

                fused[chunk_id] = dict(
                    result
                )

            fused[chunk_id][
                "rrf_score"
            ] = (
                fused[chunk_id].get(
                    "rrf_score",
                    0.0
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk_id = int(
                result["id"]
            )

            if chunk_id not in fused:

                fused[chunk_id] = dict(
                    result
                )

            fused[chunk_id][
                "rrf_score"
            ] = (
                fused[chunk_id].get(
                    "rrf_score",
                    0.0
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        results = sorted(
            fused.values(),
            key=lambda item:
                item.get(
                    "rrf_score",
                    0.0
                ),
            reverse=True,
        )

        return results[:top_k]

    # =============================================================
    # PUBLIC SEARCH
    # =============================================================

    def search_standards(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        top_k = max(int(top_k), 1)

        # ---------------------------------------------------------
        # Hybrid candidate retrieval
        #
        # Keep FAISS + BM25 + RRF as the core retrieval architecture.
        # Retrieve a larger candidate pool first, then rerank it.
        # ---------------------------------------------------------

        candidate_k = max(top_k * 5, 20)

        semantic_results = self._semantic_search(
            query,
            candidate_k
        )

        bm25_results = self._bm25_search(
            query,
            candidate_k
        )

        fused_results = self._rrf_fusion(
            semantic_results,
            bm25_results,
            candidate_k
        )

        if not fused_results:
            return []

        # ---------------------------------------------------------
        # Query-aware technical reranking
        #
        # RRF alone can favor:
        #   - standards indexes
        #   - checklist tables
        #   - document reference lists
        #   - table-of-contents pages
        #
        # For manufacturing QA we want actual technical prose.
        # ---------------------------------------------------------

        query_lower = query.lower()

        # Normalize common terminology.
        normalized_query = (
            query_lower
            .replace("ipc a 610g", "ipc-a-610g")
            .replace("ipc a-610g", "ipc-a-610g")
            .replace("ipc a 610", "ipc-a-610")
            .replace("solder bridge", "solder bridging")
        )

        query_terms = set(
            re.findall(
                r"\b[a-z0-9][a-z0-9_-]{2,}\b",
                normalized_query
            )
        )

        # Important manufacturing terms.
        technical_terms = {
            "acceptance",
            "acceptability",
            "criteria",
            "criterion",
            "acceptable",
            "unacceptable",
            "reject",
            "inspection",
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
            "class3",
            "severity",
            "electrical",
            "rework",
            "repair",
            "cause",
            "causes",
            "requirement",
            "requirements",
        }

        # Words/phrases strongly associated with navigation/index pages.
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
                item.get("text", "")
            ).strip()

            if not raw_text:
                continue

            text_lower = raw_text.lower()

            # -----------------------------------------------------
            # Base RRF score
            # -----------------------------------------------------

            rrf_score = float(
                item.get("rrf_score", 0.0)
            )

            score = rrf_score * 100.0

            # -----------------------------------------------------
            # Exact query-term overlap
            # -----------------------------------------------------

            text_terms = set(
                re.findall(
                    r"\b[a-z0-9][a-z0-9_-]{2,}\b",
                    text_lower
                )
            )

            overlap = len(
                query_terms.intersection(text_terms)
            )

            score += min(overlap * 1.5, 18.0)

            # -----------------------------------------------------
            # Technical-content bonuses
            # -----------------------------------------------------

            for term in technical_terms:
                if term in text_lower:
                    score += 1.0

            # Stronger bonuses for actual acceptance language.
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
            # Defect-specific bonuses
            # -----------------------------------------------------

            defect_phrases = [
                "solder bridging",
                "solder bridge",
                "solder short",
                "solder shorts",
                "bridging",
                "poor solder",
                "solder joint",
                "solder joints",
            ]

            for phrase in defect_phrases:
                if phrase in query_lower and phrase in text_lower:
                    score += 10.0

            # -----------------------------------------------------
            # IPC relevance
            # -----------------------------------------------------

            if "ipc-a-610g" in query_lower:

                if "ipc-a-610g" in text_lower:
                    # Direct G-revision evidence.
                    score += 30.0

                elif "ipc-a-610" in text_lower:
                    # Generic IPC-A-610 reference is useful,
                    # but NOT equivalent to IPC-A-610G.
                    score += 5.0

                if "ipc-a-610c" in text_lower:
                    # Older revision: do not let it outrank
                    # direct/current evidence.
                    score -= 8.0

            elif "ipc-a-610" in query_lower:

                if "ipc-a-610" in text_lower:
                    score += 15.0

            # -----------------------------------------------------
            # Class relevance
            # -----------------------------------------------------

            class_match = re.search(
                r"class\s*([123])",
                query_lower
            )

            if class_match:

                requested_class = class_match.group(1)

                if re.search(
                    rf"class\s*{requested_class}\b",
                    text_lower
                ):
                    score += 8.0

            # -----------------------------------------------------
            # Penalize index/checklist/reference material
            # -----------------------------------------------------

            for phrase in noise_phrases:

                if phrase in text_lower:
                    score -= 12.0

            # Tables containing many standard identifiers are
            # usually reference/index material rather than
            # acceptance criteria.
            standard_id_count = len(
                re.findall(
                    r"\b(?:ipc|j-std|ansi|iso)-?[a-z0-9-]+\b",
                    text_lower
                )
            )

            if standard_id_count >= 6:
                score -= 8.0

            if standard_id_count >= 10:
                score -= 10.0

            # -----------------------------------------------------
            # Prefer substantive technical prose.
            # -----------------------------------------------------

            sentence_count = len(
                re.findall(
                    r"[.!?]",
                    raw_text
                )
            )

            if sentence_count >= 2:
                score += 2.0

            # Extremely short fragments are less useful.
            if len(raw_text) < 120:
                score -= 3.0

            # -----------------------------------------------------
            # Save reranking diagnostics.
            # -----------------------------------------------------

            item = dict(item)

            item["retrieval_score"] = float(score)

            reranked.append(
                item
            )

        reranked.sort(
            key=lambda item:
                item.get(
                    "retrieval_score",
                    0.0
                ),
            reverse=True
        )

        # ---------------------------------------------------------
        # Return the strongest evidence.
        # ---------------------------------------------------------

        return reranked[:top_k]

    # =============================================================
    # DEFECT RETRIEVAL
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

        return self.search_standards(
            query,
            top_k
        )

    # =============================================================
    # EVIDENCE SUMMARY
    # =============================================================

    @staticmethod
    def _build_evidence_summary(
        evidence: List[
            Dict[str, Any]
        ]
    ) -> str:

        if not evidence:

            return (
                "No matching PDF evidence was retrieved."
            )

        lines = []

        for index, item in enumerate(
            evidence,
            start=1
        ):

            source = item.get(
                "source",
                "Unknown source"
            )

            page = item.get(
                "page",
                "?"
            )

            text = item.get(
                "text",
                ""
            ).strip()

            text = re.sub(
                r"\s+",
                " ",
                text
            )

            if len(text) > 500:

                text = (
                    text[:500].rstrip()
                    + "..."
                )

            lines.append(
                f"{index}. {source} "
                f"(page {page}): {text}"
            )

        return "\n".join(lines)

    # =============================================================
    # STRUCTURED STANDARDS
    # =============================================================

    @staticmethod
    def _get_structured_standard(
        defect_type: str
    ) -> Optional[
        Dict[str, Any]
    ]:

        try:

            standard = (
                get_standard_by_defect_type(
                    defect_type
                )
            )

            if standard is not None:

                return standard

        except Exception:

            pass

        normalized = (
            str(defect_type)
            .replace("_", " ")
            .strip()
            .lower()
        )

        for standard in get_all_standards():

            defect_name = str(
                standard.get(
                    "defect_type",
                    ""
                )
            ).replace(
                "_",
                " "
            ).lower()

            title = str(
                standard.get(
                    "title",
                    ""
                )
            ).lower()

            if (
                normalized == defect_name
                or normalized == title
                or normalized in defect_name
                or normalized in title
            ):

                return standard

        return None

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
                dict
            )
            else {}
        )

        clean_defect = (
            str(defect_type)
            .replace("_", " ")
            .strip()
        )

        standard = (
            self._get_structured_standard(
                defect_type
            )
        )

        defect_area_ratio = (
            visual_findings.get(
                "defect_area_ratio",
                0.0
            )
        )

        try:

            defect_area_ratio = float(
                defect_area_ratio
            )

        except (
            TypeError,
            ValueError
        ):

            defect_area_ratio = 0.0

        defect_area_ratio = max(
            0.0,
            min(
                defect_area_ratio,
                1.0
            )
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0
            )
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

        evidence = (
            self.search_standards(
                rag_query,
                top_k
            )
        )

        evidence_summary = (
            self._build_evidence_summary(
                evidence
            )
        )

        if evidence:

            knowledge_source = (
                "Manufacturer / technical PDF "
                "standards via hybrid "
                "embedding + FAISS + BM25 retrieval"
            )

        elif standard:

            knowledge_source = (
                "Structured PCB standards "
                "knowledge base (fallback)"
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
                    bool(evidence),

                "structured_kb_used":
                    bool(standard),

                "llm_generated":
                    False
            }
        }

        if standard:

            severity = standard.get(
                "severity",
                standard.get(
                    "severity_level",
                    standard.get(
                        "category",
                        "Unknown"
                    )
                )
            )

            rework_procedure = (
                standard.get(
                    "rework_procedure",
                    standard.get(
                        "ipc_7721_rework_procedure",
                        ""
                    )
                )
            )

            result["standard"] = {

                "id":
                    standard.get(
                        "id",
                        ""
                    ),

                "standard":
                    standard.get(
                        "standard",
                        ""
                    ),

                "section":
                    standard.get(
                        "section",
                        ""
                    ),

                "title":
                    standard.get(
                        "title",
                        ""
                    ),

                "category":
                    standard.get(
                        "category",
                        ""
                    ),

                "defect_type":
                    standard.get(
                        "defect_type",
                        ""
                    ),

                "description":
                    standard.get(
                        "description",
                        ""
                    ),

                "severity_level":
                    severity,

                "acceptance_level":
                    standard.get(
                        "acceptance_level",
                        ""
                    ),

                "risk_factor":
                    standard.get(
                        "risk_factor",
                        0.0
                    ),

                "electrical_impact":
                    standard.get(
                        "electrical_impact",
                        ""
                    ),

                "rework_procedure":
                    rework_procedure,

                "root_cause_factors":
                    standard.get(
                        "root_cause_factors",
                        []
                    )
            }

        else:

            result["standard"] = None

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
                                "anomaly"
                            )
                        )
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
                f"the standards document corpus."
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
                "answer": "Please provide a question regarding PCB manufacturing standards or acceptance criteria.",
                "retrieved_evidence": [],
                "evidence_count": 0,
                "operating_class": operating_class,
                "grounding": {
                    "pdf_evidence_used": False,
                    "structured_kb_used": False,
                    "llm_generated": False
                }
            }

        query_clean = query.strip()
        query_lower = query_clean.lower()

        # ---------------------------------------------------------
        # 1. Hybrid search (FAISS + BM25 + RRF)
        # ---------------------------------------------------------
        evidence = self.search_standards(query_clean, top_k)

        # ---------------------------------------------------------
        # 2. Match structured standards knowledge base
        # ---------------------------------------------------------
        kb_matches = search_standards_kb(query_clean, top_k=3)
        matched_std = kb_matches[0] if kb_matches else None

        # If hybrid evidence has a standard, check if it's more specific
        for item in evidence:
            if item.get("standard"):
                matched_std = item["standard"]
                break

        # ---------------------------------------------------------
        # 3. Comprehensive synthesis
        # ---------------------------------------------------------
        if matched_std:
            std_name = matched_std.get("standard", "IPC-A-610G")
            sec = matched_std.get("section", "")
            title = matched_std.get("title", "")
            desc = matched_std.get("description", "")
            c1 = matched_std.get("class_1_criteria", "N/A")
            c2 = matched_std.get("class_2_criteria", "N/A")
            c3 = matched_std.get("class_3_criteria", "N/A")
            acc = matched_std.get("acceptance_level", "DEFECT")
            sev = matched_std.get("severity_level", "Major")
            impact = matched_std.get("electrical_impact", "")
            rework = matched_std.get("ipc_7721_rework_procedure", "")
            causes = matched_std.get("root_cause_factors", [])

            wants_class3 = "class 3" in query_lower or operating_class == "Class 3"

            answer_parts = []
            answer_parts.append(f"### {std_name} Section {sec}: {title}\n")
            answer_parts.append(f"{desc}\n")

            answer_parts.append("**Acceptance Criteria by Product Class:**")
            answer_parts.append(f"- **Class 1 (General Electronic):** {c1}")
            answer_parts.append(f"- **Class 2 (Dedicated Service):** {c2}")
            if wants_class3:
                answer_parts.append(f"- **Class 3 (High Performance / Mission Critical):** **{c3}** *(Operational Standard)*")
            else:
                answer_parts.append(f"- **Class 3 (High Performance / Mission Critical):** {c3}")
            answer_parts.append(f"- **Compliance Level:** `{acc}` | Severity: `{sev}`\n")

            if rework and rework != "No rework required. Pass to next manufacturing stage.":
                answer_parts.append("**IPC-7711/7721 Rework & Correction Procedure:**")
                answer_parts.append(f"> {rework}\n")
            elif rework:
                answer_parts.append(f"**Rework Directive:** {rework}\n")

            if impact and impact != "None":
                answer_parts.append(f"**Reliability & Electrical Impact:** {impact}\n")

            if causes:
                answer_parts.append("**SMT Process Root Causes & Prevention:**")
                for cause in causes[:4]:
                    answer_parts.append(f"- {cause}")
                answer_parts.append("")

            # Add technical excerpt from PDF corpus if relevant
            pdf_excerpts = [item for item in evidence if item.get("source_file") != "structured_standards_kb"]
            if pdf_excerpts:
                best_pdf = pdf_excerpts[0]
                text_snippet = best_pdf.get("text", "").strip().replace("\n", " ")
                text_snippet = " ".join(text_snippet.split())
                if len(text_snippet) > 280:
                    text_snippet = text_snippet[:280].rsplit(" ", 1)[0] + "..."
                src_name = best_pdf.get("source", "Technical Guideline")
                pg = best_pdf.get("page")
                pg_str = f", Page {pg}" if pg else ""
                answer_parts.append(f"**Related Engineering Note ({src_name}{pg_str}):**")
                answer_parts.append(f"*{text_snippet}*\n")

            answer = "\n".join(answer_parts)

        elif evidence:
            best_chunk = evidence[0]
            text = best_chunk.get("text", "").strip().replace("\n", " ")
            text = " ".join(text.split())
            if len(text) > 650:
                text = text[:650].rsplit(" ", 1)[0] + "..."

            source = best_chunk.get("source", "PCB Standards Repository")
            page = best_chunk.get("page")
            pg_str = f" (Page {page})" if page else ""

            answer = (
                f"### Standards & Engineering Guideline\n\n"
                f"{text}\n\n"
                f"**Primary Source:** {source}{pg_str}"
            )

        else:
            answer = (
                "### Standards Intelligence Assistant\n\n"
                "I could not locate an exact standard matching that specific query. You can ask about:\n\n"
                "- **IPC-A-610G Acceptance Criteria**: Solder Bridging, Missing Components, Tombstoning, Solder Balls, Insufficient Solder, or Overhang\n"
                "- **Product Classifications**: Difference between Class 1, Class 2, and Class 3 reliability levels\n"
                "- **IPC-7711/7721 Rework**: Solder wick de-bridging, component replacement, touch-up temperatures\n"
                "- **ISO 9001:2015**: Clause 8.7 Non-conforming output containment and Clause 7.1.5 equipment calibration\n"
                "- **J-STD-001H**: Solder alloys, flux cleanliness, and ionic contamination testing"
            )

        # Prepare evidence list for sources display
        formatted_evidence = []
        for item in evidence[:5]:
            formatted_evidence.append({
                "source": item.get("source", "Standards Library"),
                "source_file": item.get("source_file", "Technical Guideline"),
                "page": item.get("page"),
                "text": item.get("text", "")[:250] + "..." if len(item.get("text", "")) > 250 else item.get("text", "")
            })

        return {
            "query": query_clean,
            "answer": answer,
            "retrieved_evidence": formatted_evidence,
            "evidence_count": len(formatted_evidence),
            "operating_class": operating_class,
            "grounding": {
                "pdf_evidence_used": any(item.get("source_file") != "structured_standards_kb" for item in evidence),
                "structured_kb_used": matched_std is not None,
                "llm_generated": False
            }
        }

    def list_pdf_sources(self) -> List[str]:

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

    def get_status(self) -> Dict[str, Any]:

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
                self.chunk_overlap
        }


# =============================================================
# LOCAL TEST
# =============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "ManufacturingRAG-QA â€” Standards RAG Test"
    )

    print(
        "=" * 60
    )

    engine = StandardsRAGEngine()

    print(
        "\n[RAG STATUS]"
    )

    print(
        engine.get_status()
    )

    test_query = (
        "What causes solder balls and "
        "poor solder joints during "
        "PCB assembly?"
    )

    print(
        "\n[TEST QUERY]"
    )

    print(
        test_query
    )

    results = engine.search_standards(
        test_query,
        top_k=3
    )

    print(
        f"\n[RESULTS] "
        f"{len(results)} evidence chunks\n"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            "-" * 60
        )

        print(
            f"Result {index}"
        )

        print(
            f"Source : "
            f"{result.get('source')}"
        )

        print(
            f"Page   : "
            f"{result.get('page')}"
        )

        print(
            f"RRF    : "
            f"{result.get('rrf_score', 0.0):.6f}"
        )

        print(
            f"Text   : "
            f"{result.get('text', '')[:700]}"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG test completed."
    )

    print(
        "=" * 60
    )
