"""模型、维度、接口决定向量空间；切换配置只能新建集合，不能复用旧向量。"""
import hashlib
import json


def embedding_space(embeddings):
    """无密钥指纹；同维度的不同模型仍隔离，测试替身按类型隔离。"""
    while hasattr(embeddings, "delegate"):
        embeddings = embeddings.delegate
    identity = {"model": getattr(embeddings, "model", type(embeddings).__module__ + "." + type(embeddings).__qualname__),
                "dimensions": getattr(embeddings, "dimensions", None),
                "endpoint": str(getattr(embeddings, "openai_api_base", ""))}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
