import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import settings
from custom_types import CollectionStats

logger = logging.getLogger("rag_agent.vector_db")

# Process-level client cache to prevent concurrent file-lock collisions in local storage mode
_local_client_cache: Dict[str, QdrantClient] = {}


class QdrantStorage:

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        collection: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.collection = collection or settings.QDRANT_COLLECTION_NAME
        self.dim = dim or settings.EMBEDDING_DIM
        self.mode = settings.QDRANT_MODE
        self.client = self._init_client(url=url, path=path)
        self._ensure_collection()

    def _init_client(self, url: Optional[str] = None, path: Optional[str] = None) -> QdrantClient:
        global _local_client_cache

        if self.mode == "local" or path:
            storage_path = str(Path(path or settings.QDRANT_STORAGE_PATH).resolve())
            if storage_path in _local_client_cache:
                return _local_client_cache[storage_path]
            logger.info(f"Initializing Qdrant in local embedded storage at: {storage_path}")
            client = QdrantClient(path=storage_path)
            _local_client_cache[storage_path] = client
            return client

        target_url = url or settings.QDRANT_URL
        api_key = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None

        if self.mode == "remote":
            logger.info(f"Connecting to remote Qdrant server at: {target_url}")
            return QdrantClient(url=target_url, api_key=api_key, timeout=15)

        try:
            test_client = QdrantClient(url=target_url, api_key=api_key, timeout=3)
            test_client.get_collections()
            logger.info(f"Connected to remote Qdrant server at {target_url}")
            self.mode = "remote"
            return test_client
        except Exception as e:
            storage_path = str(Path(settings.QDRANT_STORAGE_PATH).resolve())
            if storage_path in _local_client_cache:
                return _local_client_cache[storage_path]
            logger.info(f"Remote Qdrant not reachable ({e}). Falling back to local embedded disk storage: {storage_path}")
            self.mode = "local"
            client = QdrantClient(path=storage_path)
            _local_client_cache[storage_path] = client
            return client


    def _ensure_collection(self):
        try:
            if not self.client.collection_exists(self.collection):
                logger.info(f"Creating collection '{self.collection}' (dim={self.dim}, distance=Cosine)...")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
            else:
                info = self.client.get_collection(self.collection)
                existing_dim = None
                vectors_cfg = getattr(info.config.params, "vectors", None)
                if hasattr(vectors_cfg, "size"):
                    existing_dim = vectors_cfg.size
                elif isinstance(vectors_cfg, dict) and "" in vectors_cfg:
                    existing_dim = getattr(vectors_cfg[""], "size", None)

                if existing_dim and existing_dim != self.dim:
                    logger.warning(
                        f"Collection '{self.collection}' dimension mismatch (existing={existing_dim}, configured={self.dim}). "
                        f"Recreating collection for new embedding model..."
                    )
                    self.client.delete_collection(self.collection)
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                    )
        except Exception as e:
            logger.error(f"Error checking/creating collection '{self.collection}': {e}")


    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]], user_id: Optional[str] = None):
        if not ids or not vectors:
            return

        if user_id:
            for p in payloads:
                if "user_id" not in p:
                    p["user_id"] = user_id

        actual_dim = len(vectors[0])
        total = len(ids)
        batch_size = 100

        # Auto-heal: verify collection dimension against incoming vector dimension
        try:
            info = self.client.get_collection(self.collection)
            vectors_cfg = getattr(info.config.params, "vectors", None)
            coll_dim = getattr(vectors_cfg, "size", None) if hasattr(vectors_cfg, "size") else (
                getattr(vectors_cfg.get(""), "size", None) if isinstance(vectors_cfg, dict) else None
            )
            if coll_dim and coll_dim != actual_dim:
                logger.warning(
                    f"Collection '{self.collection}' has dimension {coll_dim}, but incoming vectors have dimension {actual_dim}. "
                    f"Auto-recreating collection to match new embedding model..."
                )
                self.client.delete_collection(self.collection)
                self.dim = actual_dim
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.warning(f"Dimension check pre-upsert: {e}")

        try:
            for i in range(0, total, batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]
                batch_payloads = payloads[i : i + batch_size]

                points = [
                    PointStruct(
                        id=batch_ids[j],
                        vector=batch_vectors[j],
                        payload=batch_payloads[j],
                    )
                    for j in range(len(batch_ids))
                ]
                self.client.upsert(collection_name=self.collection, points=points)
        except Exception as e:
            if "Vector dimension error" in str(e) or "dimension" in str(e).lower():
                logger.warning(f"Caught vector dimension error: {e}. Recreating collection with dim={actual_dim}...")
                self.client.delete_collection(self.collection)
                self.dim = actual_dim
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE),
                )
                for i in range(0, total, batch_size):
                    batch_ids = ids[i : i + batch_size]
                    batch_vectors = vectors[i : i + batch_size]
                    batch_payloads = payloads[i : i + batch_size]
                    points = [
                        PointStruct(id=batch_ids[j], vector=batch_vectors[j], payload=batch_payloads[j])
                        for j in range(len(batch_ids))
                    ]
                    self.client.upsert(collection_name=self.collection, points=points)
            else:
                raise

        logger.info(f"Successfully upserted {total} vectors into '{self.collection}' (user_id={user_id}).")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            actual_dim = len(query_vector)
            info = self.client.get_collection(self.collection)
            vectors_cfg = getattr(info.config.params, "vectors", None)
            coll_dim = getattr(vectors_cfg, "size", None) if hasattr(vectors_cfg, "size") else None
            if coll_dim and coll_dim != actual_dim:
                logger.warning(
                    f"Search vector dim ({actual_dim}) does not match collection dim ({coll_dim}). Recreating collection..."
                )
                self.client.delete_collection(self.collection)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE),
                )
                return {"contexts": [], "sources": [], "citations": []}

            # Multi-tenant user isolation filter
            query_filter = None
            if user_id:
                query_filter = Filter(
                    should=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="user_id", match=MatchValue(value="public")),
                    ]
                )

            # Compatible with both qdrant-client search and query_points
            if hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
            else:
                query_res = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    with_payload=True,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                results = query_res.points
        except Exception as e:
            logger.error(f"Search query error: {e}")
            return {"contexts": [], "sources": [], "citations": []}

        contexts: List[str] = []
        sources: set = set()
        citations: List[Dict[str, Any]] = []

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "unknown")
            score = getattr(r, "score", None)
            chunk_index = payload.get("chunk_index", None)
            point_user_id = payload.get("user_id")

            if text:
                contexts.append(text)
                sources.add(source)
                citations.append({
                    "text": text,
                    "source": source,
                    "score": round(float(score), 4) if score is not None else None,
                    "chunk_index": chunk_index,
                    "user_id": point_user_id,
                })

        return {
            "contexts": contexts,
            "sources": list(sources),
            "citations": citations,
        }

    def get_stats(self, user_id: Optional[str] = None) -> CollectionStats:
        try:
            info = self.client.get_collection(self.collection)
            total_vectors = getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0) or 0
            sources = self.list_sources(user_id=user_id)
            return CollectionStats(
                collection_name=self.collection,
                status="healthy",
                mode=self.mode,
                total_vectors=total_vectors,
                sources=sources,
            )
        except Exception as e:
            logger.warning(f"Unable to retrieve stats for '{self.collection}': {e}")
            return CollectionStats(
                collection_name=self.collection,
                status="unreachable",
                mode=self.mode,
                total_vectors=0,
                sources=[],
            )

    def list_sources(self, user_id: Optional[str] = None) -> List[str]:
        try:
            scroll_filter = None
            if user_id:
                scroll_filter = Filter(
                    should=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="user_id", match=MatchValue(value="public")),
                    ]
                )

            scroll_result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            points, _ = scroll_result
            unique_sources = set()
            for p in points:
                payload = getattr(p, "payload", None) or {}
                src = payload.get("source")
                if src:
                    unique_sources.add(src)
            return sorted(list(unique_sources))
        except Exception:
            return []

    def delete_source(self, source_id: str, user_id: Optional[str] = None) -> bool:
        try:
            conditions = [
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_id),
                )
            ]
            if user_id:
                conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                )
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(must=conditions),
            )
            logger.info(f"Deleted points for source '{source_id}' (user_id={user_id}) from '{self.collection}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete source '{source_id}': {e}")
            return False