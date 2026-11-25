import asyncio
from unittest.mock import Mock

import httpx
from fastapi import FastAPI

from api.chat import router
from api.deps import get_chat_service, get_limited_user
from utils.security import create_token
from memory.session_lease import SessionLeaseStore
from services.chat_service import ChatService


def make_app(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_service] = lambda: service
    app.dependency_overrides[get_limited_user] = lambda: "pytest_user"
    return app


def test_invalid_requests_are_rejected_before_agent():
    async def run():
        service = Mock()
        app = make_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for body in [
                {"message": "你好"},
                {"message": " ", "session_id": "s"},
                {"message": "你好", "session_id": ""},
                {"message": "你好", "session_id": "a:b"},
                {"message": "字" * 4001, "session_id": "s"},
            ]:
                response = await client.post("/chat", json=body)
                assert response.status_code == 422
        service.open_turn.assert_not_called()
    asyncio.run(run())


def test_busy_session_returns_http_409(repository):
    async def run():
        service = ChatService(Mock(), SessionLeaseStore(), repository)
        turn = await service.open_turn("s", "第一问", "pytest_user")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(service)), base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "第二问", "session_id": "s"})
            assert response.status_code == 409
        await service.close_turn(turn)
    asyncio.run(run())


def test_stream_failure_sends_error_and_done(repository):
    class FailedAgent:
        async def execute_stream(self, messages, user_id):
            raise RuntimeError("internal secret")
            yield

    async def run():
        service = ChatService(FailedAgent(), SessionLeaseStore(), repository)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(service)), base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "你好", "session_id": "s"})
            assert response.status_code == 200
            assert '"type": "error"' in response.text
            assert "[DONE]" in response.text
            assert "internal secret" not in response.text
    asyncio.run(run())


def test_jwt_identity_scopes_real_http_requests(repository):
    class Agent:
        async def execute_stream(self, messages, user_id):
            yield {"type": "answer", "content": f"当前身份 {user_id}"}

    async def run():
        service = ChatService(Agent(), SessionLeaseStore(), repository)
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_chat_service] = lambda: service
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.post("/chat", json={"message": "你好", "session_id": "same"})
            assert denied.status_code in (401, 403)
            for user_id in ("1001", "1002"):
                response = await client.post(
                    "/chat", json={"message": f"秘密 {user_id}", "session_id": "same"},
                    headers={"Authorization": f"Bearer {create_token(user_id)}"},
                )
                assert response.status_code == 200
        first = repository.history(service.session_scope("1001", "same"), "1001")
        second = repository.history(service.session_scope("1002", "same"), "1002")
        assert "1002" not in str(first)
        assert "1001" not in str(second)
    asyncio.run(run())
