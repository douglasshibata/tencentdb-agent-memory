import logging
import math
import hashlib
import json
from typing import List, Dict, Any, Optional
from src.config import settings

logger = logging.getLogger(__name__)

# Try to import tcvectordb, fall back to mock if not installed or mock mode is active
TENCENT_SDK_AVAILABLE = False
tcvectordb_sdk: Any = None

if not settings.MOCK_MODE:
    try:
        import tcvectordb
        TENCENT_SDK_AVAILABLE = True
        tcvectordb_sdk = tcvectordb
    except ImportError:
        logger.warning(
            "tcvectordb Python SDK not installed. Falling back to Mock Vector DB implementation."
        )
        TENCENT_SDK_AVAILABLE = False

# Fallback/Mock classes to replicate Tencent VectorDB SDK structure
class MockFieldType:
    String = "String"
    Uint64 = "Uint64"
    Vector = "Vector"

class MockIndexType:
    PRIMARY_KEY = "PRIMARY_KEY"
    FILTER = "FILTER"
    HNSW = "HNSW"

class MockMetricType:
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"

class MockReadConsistency:
    EVENTUAL = "EVENTUAL"
    STRONG = "STRONG"

class MockFilterIndex:
    def __init__(self, name: str, field_type: str, index_type: str):
        self.name = name
        self.field_type = field_type
        self.index_type = index_type

class MockVectorIndex:
    def __init__(self, name: str, dimension: int, index_type: str, metric_type: str, params: Any = None):
        self.name = name
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.params = params

class MockIndex:
    def __init__(self, *args: Any):
        self.indexes = list(args)

    def add(self, index_field: Any):
        self.indexes.append(index_field)

class MockDocument:
    def __init__(self, id: str, vector: List[float], **kwargs: Any):
        self.id = id
        self.vector = vector
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "vector": self.vector}
        for k, v in self.__dict__.items():
            if k not in ["id", "vector"]:
                d[k] = v
        return d

class MockFilter:
    def __init__(self, filter_expr: str):
        self.filter_expr = filter_expr

    @staticmethod
    def In(field_name: str, values: List[Any]):
        return f"{field_name} in {json.dumps(values)}"


# Define local memory database for mock storage
_MOCK_DATABASES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

class MockCollection:
    def __init__(self, db_name: str, col_name: str, index: Any = None, description: str = ""):
        self.db_name = db_name
        self.collection_name = col_name
        self.index = index
        self.description = description
        if db_name not in _MOCK_DATABASES:
            _MOCK_DATABASES[db_name] = {}
        if col_name not in _MOCK_DATABASES[db_name]:
            _MOCK_DATABASES[db_name][col_name] = []

    def upsert(self, documents: List[Any], build_index: bool = True, timeout: int = 10) -> Dict[str, Any]:
        logger.info(f"[MockDB {self.db_name}/{self.collection_name}] Upserting {len(documents)} documents.")
        store = _MOCK_DATABASES[self.db_name][self.collection_name]
        upserted_ids = []
        for doc in documents:
            doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else doc.__dict__
            # Clean duplicate ID
            store[:] = [d for d in store if d.get("id") != doc_dict.get("id")]
            store.append(doc_dict)
            upserted_ids.append(doc_dict.get("id"))
        return {"code": 0, "msg": "success", "upserted_count": len(upserted_ids)}

    def query(self, document_ids: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        store = _MOCK_DATABASES[self.db_name][self.collection_name]
        results = [d for d in store if d.get("id") in document_ids]
        return results[:limit]

    def delete(self, filter: Any = None, document_ids: List[str] = None) -> Dict[str, Any]:
        store = _MOCK_DATABASES[self.db_name][self.collection_name]
        initial_len = len(store)
        if document_ids:
            store[:] = [d for d in store if d.get("id") not in document_ids]
        elif filter:
            filter_str = str(filter.filter_expr) if hasattr(filter, "filter_expr") else str(filter)
            if "id in" in filter_str:
                # Extract ids
                try:
                    import re
                    match = re.search(r"\[.*\]", filter_str)
                    if match:
                        ids_to_del = json.loads(match.group(0))
                        store[:] = [d for d in store if d.get("id") not in ids_to_del]
                except Exception as e:
                    logger.error(f"Error parsing mock delete filter: {e}")
            elif "user_id" in filter_str:
                # Mock delete by user_id
                try:
                    import re
                    match = re.search(r'user_id\s*=\s*"([^"]+)"', filter_str)
                    if match:
                        uid = match.group(1)
                        store[:] = [d for d in store if d.get("user_id") != uid]
                except Exception as e:
                    logger.error(f"Error parsing user_id delete filter: {e}")
        return {"code": 0, "msg": "success", "deleted_count": initial_len - len(store)}

    def search(
        self,
        vectors: List[List[float]],
        limit: int = 10,
        retrieve_vector: bool = True,
        output_fields: Optional[List[str]] = None,
        filter: Any = None
    ) -> List[List[Dict[str, Any]]]:
        store = _MOCK_DATABASES[self.db_name][self.collection_name]
        if not vectors:
            return [[]]

        query_vec = vectors[0]
        scored_results = []

        # Parse simple filters
        filter_user_id = None
        filter_session_id = None
        if filter:
            filter_str = str(filter.filter_expr) if hasattr(filter, "filter_expr") else str(filter)
            import re
            u_match = re.search(r'user_id\s*=\s*"([^"]+)"', filter_str)
            if u_match:
                filter_user_id = u_match.group(1)
            s_match = re.search(r'session_id\s*=\s*"([^"]+)"', filter_str)
            if s_match:
                filter_session_id = s_match.group(1)

        for doc in store:
            # Apply filters
            if filter_user_id and doc.get("user_id") != filter_user_id:
                continue
            if filter_session_id and doc.get("session_id") != filter_session_id:
                continue

            doc_vec = doc.get("vector", [])
            similarity = self._cosine_similarity(query_vec, doc_vec)

            # Format output document
            doc_out = {
                "id": doc.get("id"),
                "score": similarity,
                "text": doc.get("text"),
                "user_id": doc.get("user_id"),
                "session_id": doc.get("session_id"),
                "timestamp": doc.get("timestamp"),
                "metadata": doc.get("metadata", "{}")
            }
            if retrieve_vector:
                doc_out["vector"] = doc_vec

            scored_results.append(doc_out)

        # Sort by similarity score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return [scored_results[:limit]]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

class MockDatabase:
    def __init__(self, database_name: str):
        self.database_name = database_name

    def describe_collection(self, collection_name: str) -> MockCollection:
        return MockCollection(self.database_name, collection_name)

    def create_collection(self, name: str, shard: int, replicas: int, description: str, index: Any) -> MockCollection:
        return MockCollection(self.database_name, name, index, description)

    def drop_collection(self, collection_name: str):
        if self.database_name in _MOCK_DATABASES and collection_name in _MOCK_DATABASES[self.database_name]:
            del _MOCK_DATABASES[self.database_name][collection_name]
        return {"code": 0, "msg": "success"}

class MockVectorDBClient:
    def __init__(self, url: str, username: str, key: str, read_consistency: Any = None, timeout: int = 10):
        self.url = url
        self.username = username
        self.key = key
        self.read_consistency = read_consistency
        self.timeout = timeout

    def list_databases(self) -> List[MockDatabase]:
        return [MockDatabase(name) for name in _MOCK_DATABASES.keys()]

    def database(self, database_name: str) -> MockDatabase:
        if database_name not in _MOCK_DATABASES:
            _MOCK_DATABASES[database_name] = {}
        return MockDatabase(database_name)

    def create_database(self, database_name: str) -> MockDatabase:
        if database_name not in _MOCK_DATABASES:
            _MOCK_DATABASES[database_name] = {}
        return MockDatabase(database_name)


class EmbeddingGenerator:
    """Deterministic embedding generator yielding realistic unit-length 1536-dim vectors for testing."""
    @staticmethod
    def generate(text: str, dimension: int = 1536) -> List[float]:
        if settings.EMBEDDING_PROVIDER == "openai" and not settings.MOCK_MODE:
            try:
                import openai
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                resp = client.embeddings.create(input=[text], model=settings.EMBEDDING_MODEL)
                return resp.data[0].embedding
            except Exception as e:
                logger.error(f"Error calling real OpenAI embedding API: {e}. Falling back to deterministic mock.")

        # Real Tencent Hunyuan Embeddings API could go here

        # Pure Python deterministic embeddings generator
        # Map letters to dimensions to simulate overlap semantic similarity
        vector = [0.0] * dimension
        words = text.lower().split()

        # Set indices based on word hashes to create semantic similarity for overlapping words
        for idx, word in enumerate(words):
            # Generate deterministic hashes for words
            h = hashlib.sha256(word.encode("utf-8")).hexdigest()
            pos = int(h, 16) % dimension
            # Set values
            vector[pos] += 1.0
            # Also spread energy to neighbors to simulate fuzzy similarity
            vector[(pos + 1) % dimension] += 0.5
            vector[(pos - 1) % dimension] += 0.5

        # Always add overall sentence hash component
        h_full = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        for offset in range(5):
            pos = (int(h_full, 16) + offset * 7) % dimension
            vector[pos] += 0.2

        # Normalize to unit length (L2 norm) so cosine similarity = dot product
        sq_sum = sum(val * val for val in vector)
        if sq_sum == 0.0:
            # Fallback if text is empty
            vector[0] = 1.0
            return vector

        norm = math.sqrt(sq_sum)
        return [val / norm for val in vector]


class TencentVectorConnector:
    """Adapter for connecting to and interacting with Tencent Cloud Vector Database."""
    def __init__(self):
        self.mock_mode = settings.MOCK_MODE or not TENCENT_SDK_AVAILABLE
        self.host = settings.TENCENTDB_HOST
        self.port = settings.TENCENTDB_PORT
        self.username = settings.TENCENTDB_USERNAME
        self.key = settings.TENCENTDB_KEY
        self.database_name = settings.TENCENTDB_DATABASE_NAME
        self.collection_name = settings.TENCENTDB_COLLECTION_NAME

        self._client = None
        self._db = None
        self._collection = None

        self._initialize_client()

    def _initialize_client(self):
        try:
            if self.mock_mode:
                logger.info("Initializing Mock Tencent VectorDB Client...")
                self._client = MockVectorDBClient(
                    url=f"http://{self.host}:{self.port}",
                    username=self.username,
                    key=self.key
                )
                self._db = self._client.create_database(self.database_name)

                # Mock Index structure setup
                index = MockIndex(
                    MockFilterIndex("id", MockFieldType.String, MockIndexType.PRIMARY_KEY),
                    MockFilterIndex("user_id", MockFieldType.String, MockIndexType.FILTER),
                    MockFilterIndex("session_id", MockFieldType.String, MockIndexType.FILTER),
                    MockVectorIndex("vector", 1536, MockIndexType.HNSW, MockMetricType.COSINE)
                )
                self._collection = self._db.create_collection(
                    name=self.collection_name,
                    shard=1,
                    replicas=0,
                    description="AI Agent Memory Collection",
                    index=index
                )
            else:
                logger.info(f"Connecting to live Tencent VectorDB at {self.host}:{self.port}...")
                import tcvectordb
                from tcvectordb.model.enum import ReadConsistency, FieldType, IndexType, MetricType
                from tcvectordb.model.index import Index, FilterIndex, VectorIndex

                self._client = tcvectordb.VectorDBClient(
                    url=f"http://{self.host}:{self.port}",
                    username=self.username,
                    key=self.key,
                    read_consistency=ReadConsistency.STRONG,
                    timeout=15
                )

                # Ensure DB exists
                db_list = self._client.list_databases()
                if self.database_name in [d.database_name for d in db_list]:
                    self._db = self._client.database(self.database_name)
                else:
                    self._db = self._client.create_database(self.database_name)

                # Ensure Collection exists
                try:
                    self._collection = self._db.describe_collection(self.collection_name)
                except Exception:
                    # Create Collection with proper indexes
                    index = Index(
                        FilterIndex("id", FieldType.String, IndexType.PRIMARY_KEY),
                        FilterIndex("user_id", FieldType.String, IndexType.FILTER),
                        FilterIndex("session_id", FieldType.String, IndexType.FILTER),
                        VectorIndex("vector", 1536, IndexType.HNSW, MetricType.COSINE)
                    )
                    self._collection = self._db.create_collection(
                        name=self.collection_name,
                        shard=1,
                        replicas=0,
                        description="AI Agent Memory Collection",
                        index=index
                    )
                logger.info("Successfully connected to Tencent VectorDB collection.")
        except Exception as e:
            logger.error(f"Failed to initialize Tencent VectorDB client: {e}. Switching to Mock fallback.")
            self.mock_mode = True
            # Re-run mock setup
            self._initialize_client()

    def add_document(self, id: str, text: str, user_id: str, session_id: str, timestamp: float, metadata: Dict[str, Any]) -> bool:
        """Upsert a single memory document into TencentDB."""
        try:
            vector = EmbeddingGenerator.generate(text)

            if self.mock_mode:
                doc = MockDocument(
                    id=id,
                    vector=vector,
                    text=text,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=timestamp,
                    metadata=json.dumps(metadata)
                )
                res = self._collection.upsert(documents=[doc])
                return res.get("code") == 0
            else:
                from tcvectordb.model.document import Document
                doc = Document(
                    id=id,
                    vector=vector,
                    text=text,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=timestamp,
                    metadata=json.dumps(metadata)
                )
                res = self._collection.upsert(documents=[doc])
                return True
        except Exception as e:
            logger.error(f"Error adding document {id}: {e}")
            return False

    def search_similarity(self, query_text: str, user_id: str, session_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search with meta filters over user/session dimensions."""
        try:
            query_vector = EmbeddingGenerator.generate(query_text)

            # Construct standard filtering expression
            filter_exprs = [f'user_id = "{user_id}"']
            if session_id:
                filter_exprs.append(f'session_id = "{session_id}"')
            filter_expr = " and ".join(filter_exprs)

            if self.mock_mode:
                filter_obj = MockFilter(filter_expr)
                results_batch = self._collection.search(
                    vectors=[query_vector],
                    limit=limit,
                    retrieve_vector=True,
                    filter=filter_obj
                )
            else:
                from tcvectordb.model.document import Filter
                filter_obj = Filter(filter_expr)
                results_batch = self._collection.search(
                    vectors=[query_vector],
                    limit=limit,
                    retrieve_vector=True,
                    filter=filter_obj
                )

            formatted_results = []
            if results_batch and len(results_batch) > 0:
                for doc in results_batch[0]:
                    metadata_raw = doc.get("metadata", "{}")
                    parsed_meta = {}
                    if isinstance(metadata_raw, str):
                        try:
                            parsed_meta = json.loads(metadata_raw)
                        except Exception:
                            parsed_meta = {}
                    elif isinstance(metadata_raw, dict):
                        parsed_meta = metadata_raw

                    formatted_results.append({
                        "id": doc.get("id"),
                        "text": doc.get("text"),
                        "score": doc.get("score"),
                        "user_id": doc.get("user_id"),
                        "session_id": doc.get("session_id"),
                        "timestamp": doc.get("timestamp"),
                        "metadata": parsed_meta
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def delete_document(self, id: str) -> bool:
        """Delete a single document by unique identifier."""
        try:
            if self.mock_mode:
                res = self._collection.delete(document_ids=[id])
                return True
            else:
                from tcvectordb.model.document import Filter
                res = self._collection.delete(document_ids=[id])
                return True
        except Exception as e:
            logger.error(f"Error deleting document {id}: {e}")
            return False

    def delete_by_user(self, user_id: str) -> bool:
        """Delete all documents belonging to a user ID."""
        try:
            filter_expr = f'user_id = "{user_id}"'
            if self.mock_mode:
                filter_obj = MockFilter(filter_expr)
                self._collection.delete(filter=filter_obj)
            else:
                from tcvectordb.model.document import Filter
                filter_obj = Filter(filter_expr)
                self._collection.delete(filter=filter_obj)
            return True
        except Exception as e:
            logger.error(f"Error deleting user {user_id} documents: {e}")
            return False
