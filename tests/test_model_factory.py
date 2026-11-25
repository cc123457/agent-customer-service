"""校验提供方批量限制，防止真实索引首次入库时才出现400错误。"""
from types import SimpleNamespace


def test_embedding_sdk_splits_large_input_without_losing_vectors():
    """真实SDK须将21条文本分成不超过10条的批次，返回顺序与数量仍完整。"""
    from model.factory import embed_model
    batches = []

    def create(input, **kwargs):
        """模拟服务端的10条上限；不发送真实请求，只替换SDK最末端HTTP客户端。"""
        assert len(input) <= 10
        batches.append(list(input))
        return {"data": [{"embedding": [float(text), 1.0]} for text in input]}

    model = embed_model.model_copy(update={"client": SimpleNamespace(create=create)})
    vectors = model.embed_documents([str(index) for index in range(21)])
    assert [len(batch) for batch in batches] == [10, 10, 1]
    assert [vector[0] for vector in vectors] == list(range(21))
