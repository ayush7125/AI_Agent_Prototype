from sentence_transformers import SentenceTransformer
import chromadb
from utils import chunk_text
import os

EMBED_MODEL = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, persist_dir='db/chroma'):
        # ✅ Ensure directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # ✅ Use the new Chroma PersistentClient API
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None
        self.embed = SentenceTransformer(EMBED_MODEL)

    def create_collection(self, name='scholarmind'):
        # ✅ If collection exists, load it instead of recreating
        existing = [col.name for col in self.client.list_collections()]
        if name in existing:
            print(f"📁 Using existing Chroma collection: {name}")
            self.collection = self.client.get_collection(name)
        else:
            print(f"📁 Creating new Chroma collection: {name}")
            self.collection = self.client.create_collection(name)

    def index_documents(self, docs: list):
        """
        docs: list of dicts {id, text, metadata}
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        texts = [d['text'] for d in docs]
        embeddings = self.embed.encode(texts, show_progress_bar=True).tolist()
        ids = [d['id'] for d in docs]
        metadatas = [d.get('metadata', {}) for d in docs]

        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        print(f"✅ Indexed {len(docs)} documents.")

    def query(self, query_text: str, k=5):
        """
        Returns top-k most relevant documents for a query.
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        emb = self.embed.encode([query_text]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        out = []
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            out.append({'text': doc, 'meta': meta, 'distance': dist})
        return out


if __name__ == '__main__':
    r = Retriever()
    r.create_collection()

    # Example indexing
    docs = [
        {'id': 'd1', 'text': 'This is a sample doc about deep learning.', 'metadata': {'title': 'sample'}}
    ]
    r.index_documents(docs)

    # Example query
    print(r.query('deep learning'))
