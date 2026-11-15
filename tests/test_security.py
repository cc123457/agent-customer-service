from utils.security import create_token, decode_token, hash_password, verify_password


def test_hash_and_verify():
    """正确密码校验通过，错误密码校验失败"""
    stored = hash_password("123456")
    assert verify_password("123456", stored) is True
    assert verify_password("wrong", stored) is False


def test_salt_makes_hash_different():
    """相同密码因随机盐不同，哈希也不同（防彩虹表）"""
    h1 = hash_password("123456")
    h2 = hash_password("123456")
    assert h1 != h2


def test_create_and_decode_token():
    """签发的 token 能解析出正确的用户 ID"""
    token = create_token("1001")
    payload = decode_token(token)
    assert payload["sub"] == "1001"
