from memory import (
    BM25Retriever,
    HashEmbeddingModel,
    HybridRetriever,
    InMemoryMemoryStore,
    MemoryRecord,
    VectorRetriever,
    WeightedReranker,
)


def _store():
    store = InMemoryMemoryStore()
    store.add(MemoryRecord(memory_id="db", content="项目数据库使用 openGauss。", entities=["openGauss"], importance=0.9))
    store.add(MemoryRecord(memory_id="arch", content="最终部署架构采用 ARM64。", entities=["ARM64"], importance=0.8))
    store.add(MemoryRecord(memory_id="noise", content="今天下午天气很好。", importance=0.1))
    return store


def test_vector_and_hybrid_retrieval_v1():
    store = _store()
    embedder = HashEmbeddingModel(dim=256)
    vector = VectorRetriever(store, embedder)
    bm25 = BM25Retriever(store)
    hybrid = HybridRetriever(vector, bm25)
    hits = hybrid.search("项目用的 openGauss 数据库是什么？", top_k=2)
    assert hits[0].memory_id == "db"
    reranked = WeightedReranker(store).rerank("项目用的 openGauss 数据库是什么？", hits, top_k=1)
    assert reranked[0].memory_id == "db"
    assert "features" in reranked[0].metadata
