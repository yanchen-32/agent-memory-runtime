from memory import HashEmbeddingModel, VectorMemoryStore


def test_vector_store_retrieves_related_memory():
    store = VectorMemoryStore(HashEmbeddingModel(dim=256))
    target_id = store.add("项目最终部署架构是 ARM64。", memory_id="target")
    store.add("今天下午天气很好。", memory_id="noise")
    hits = store.search("最终部署架构是什么？", top_k=1)
    assert target_id == "target"
    assert hits[0].memory_id == "target"
