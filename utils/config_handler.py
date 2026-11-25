"""
yaml
k: v
"""
import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    """读取指定 YAML 并返回 RAG/聊天模型配置，默认加载 config/rag.yml；文件错误向外抛。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    """读取并返回 Chroma 路径、分片与召回参数，供 VectorStoreService 使用，不初始化向量库。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    """读取并返回提示词文件路径配置，供 prompt_loader 加载正文，不调用模型。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    """读取并返回 Agent 相关 YAML 配置；路径/编码可指定，加载失败不静默提供默认值。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
with open(get_abs_path("config/memory.yml"), encoding="utf-8") as memory_config_file:
    memory_conf = yaml.safe_load(memory_config_file)


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
