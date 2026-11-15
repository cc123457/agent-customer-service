from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.repository import RecordRepository
from utils.security import create_token, verify_password

router = APIRouter()

repo = RecordRepository()


class LoginRequest(BaseModel):
    """登录请求体"""
    username: str   # demo 中直接用用户 ID 作为登录名
    password: str


@router.post("/login")
def login(req: LoginRequest):
    """登录：校验用户名密码，签发 JWT"""
    user = repo.get_user_by_id(req.username)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_token(user["user_id"])
    return {"access_token": token, "token_type": "bearer"}
