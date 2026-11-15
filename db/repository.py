"""数据访问层：封装 SQLite 读写，隔离 SQL 细节。

采用标准库 sqlite3（零依赖）。将来若迁移 MySQL，只需替换本文件的连接方式与 SQL 方言，
调用方（工具层）无感知。
"""
import csv
import sqlite3
from datetime import datetime

from utils.path_tool import get_abs_path
from utils.security import hash_password

DB_PATH = get_abs_path("data/agent.db")


def get_conn() -> sqlite3.Connection:
    """获取数据库连接；row_factory 让查询结果支持按列名访问"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表并从 CSV 导入数据（幂等：重复执行结果一致）"""
    conn = get_conn()
    cur = conn.cursor()

    # 重建表（DROP + CREATE）：表结构会演进（如新增 password_hash），重建保证干净
    cur.execute("DROP TABLE IF EXISTS users")
    cur.execute("DROP TABLE IF EXISTS usage_records")

    cur.execute("""
        CREATE TABLE users (
            user_id       TEXT PRIMARY KEY,
            city          TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE usage_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT NOT NULL,
            month        TEXT NOT NULL,
            feature      TEXT,
            efficiency   TEXT,
            consumables  TEXT,
            comparison   TEXT
        )
    """)

    # 城市映射：CSV 里没有城市字段，这里补充（真实系统应来自用户档案表）
    city_map = {
        "1001": "深圳", "1002": "北京", "1003": "上海", "1004": "杭州",
        "1005": "广州", "1006": "成都", "1007": "武汉", "1008": "南京",
        "1009": "西安", "1010": "重庆",
    }
    # demo 预设所有用户统一密码 123456；每个用户单独 hash（盐随机，故哈希各不相同）
    for uid, city in city_map.items():
        cur.execute(
            "INSERT INTO users (user_id, city, password_hash) VALUES (?, ?, ?)",
            (uid, city, hash_password("123456")),
        )

    # 导入使用记录，并把月份平移到当前年份（数据是历史快照，平移后「当前月」才能查到）
    current_year = datetime.now().year
    csv_path = get_abs_path("data/external/records.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            user_id, feature, efficiency, consumables, comparison, month = row
            new_month = f"{current_year}-{month.split('-')[1]}"
            cur.execute(
                "INSERT INTO usage_records "
                "(user_id, month, feature, efficiency, consumables, comparison) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, new_month, feature, efficiency, consumables, comparison),
            )

    conn.commit()
    conn.close()


class RecordRepository:
    """用户使用记录的数据访问封装"""

    def get_record(self, user_id: str, month: str) -> dict | None:
        """查询指定用户、指定月份的使用记录"""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM usage_records WHERE user_id = ? AND month = ?",
            (user_id, month),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "特征": row["feature"],
            "效率": row["efficiency"],
            "耗材": row["consumables"],
            "对比": row["comparison"],
        }

    def get_user_ids(self) -> list[str]:
        """返回全部用户 ID"""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users")
        rows = cur.fetchall()
        conn.close()
        return [r["user_id"] for r in rows]

    def get_user_city(self, user_id: str) -> str | None:
        """查询指定用户所在城市"""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT city FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return row["city"] if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        """查询完整用户信息（含密码哈希），供登录校验使用"""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "city": row["city"],
            "password_hash": row["password_hash"],
        }
