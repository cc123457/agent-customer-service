"""建库脚本：初始化 SQLite 数据库并导入示例数据

用法（在项目根目录执行）：
    python -m scripts.init_db
"""
from db.repository import init_db

if __name__ == "__main__":
    init_db()
    print("数据库初始化完成：data/agent.db")
