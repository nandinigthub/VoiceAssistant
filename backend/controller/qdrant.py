from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from ..config import QDRANT_COLLECTION
client = QdrantClient(host="localhost", port=6333)

class VectorStore:
    def __init__(self,collection: str = QDRANT_COLLECTION):
        self.client = client
        self.collection = collection
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

    def add_documents(self, docs):
        texts = []
        points = []
        for idx, doc in enumerate(docs):
            text = doc.page_content
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id", idx)
        
            texts.append(text)
            
            # embeddings
            vector = self.embeddings.embed_documents(text,chunk_size=512)
            payload = {"text": text, "meta": meta}
            points.append(models.PointStruct(id=doc_id, vector=vector[0], payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int = 3,score_threshold:float=0.7):
        print("search")
        emb = self.embeddings.embed_query(query)
        hits = self.client.search(collection_name=self.collection, query_vector=emb, limit=top_k,score_threshold=score_threshold)
        results = []
        for h in hits:
            results.append({"id": h.id, "text": h.payload.get("text", ""), "score": float(h.score)})
        return results
