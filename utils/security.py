"""安全工具：密码哈希 + JWT 签发/验证"""
import hashlib
import os
import time

import jwt
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

# 显式加载项目根的 .env，确保无论从哪个目录启动都能读到密钥
load_dotenv(get_abs_path(".env"), override=True)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")  # 生产环境必须配置
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60   # token 有效期（分钟）


def hash_password(password: str, salt: bytes = None) -> str:
    """用 PBKDF2 对密码做加盐哈希，返回 '盐$哈希' 字符串。

    加盐：即使两个用户密码相同，随机盐不同，哈希也不同，防彩虹表攻击。
    """
    if salt is None:
        salt = os.urandom(16)  # 16 字节随机盐
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码是否与存储的哈希匹配"""
    try:
        salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return dk.hex() == hash_hex


def create_token(user_id: str) -> str:
    """签发 JWT：payload 里放用户 ID（sub）和过期时间（exp）"""
    payload = {
        "sub": user_id,                                       # subject，标准字段放用户标识
        "exp": int(time.time()) + TOKEN_EXPIRE_MINUTES * 60,  # 过期时间（Unix 秒）
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """验证并解析 JWT，返回 payload；无效或过期会抛异常"""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
