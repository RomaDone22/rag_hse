import os
import numpy as np
import faiss
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType

USE_MILVUS = True
MILVUS_COLLECTION_NAME = "documents"
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

doc_texts = []
doc_index = None
milvus_collection = None


def init_retriever(doc_embeddings):
    global doc_index, milvus_collection

    if USE_MILVUS:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(MILVUS_COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            ]
            schema = CollectionSchema(fields)
            milvus_collection = Collection(name=MILVUS_COLLECTION_NAME, schema=schema)
            milvus_collection.create_index("embedding", {
                "index_type": "HNSW", "metric_type": "IP", "params": {"M": 8, "efConstruction": 64}
            })
            ids = list(range(len(doc_embeddings)))
            milvus_collection.insert([ids, doc_embeddings.tolist()])
            milvus_collection.load()
        else:
            milvus_collection = Collection(name=MILVUS_COLLECTION_NAME)
            milvus_collection.load()
    else:
        dim = doc_embeddings.shape[1]
        doc_index = faiss.IndexFlatIP(dim)
        doc_index.add(doc_embeddings)


def search_documents(query_embedding, top_k=5):
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    if USE_MILVUS:
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = milvus_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )
        hits = results[0]
        return [(doc_texts[int(hit.id)], hit.distance) for hit in hits]
    else:
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = doc_index.search(query_embedding, top_k)
        return [(doc_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]