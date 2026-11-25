import json
import uuid

import requests
import streamlit as st

# 后端服务地址
BACKEND_URL = "http://127.0.0.1:8000"

st.title("智扫通机器人智能客服")
st.divider()

# 初始化会话状态
if "token" not in st.session_state:
    st.session_state["token"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_request" not in st.session_state:
    st.session_state["pending_request"] = None


# ========== 登录区 ==========
if not st.session_state["token"]:
    st.subheader("请先登录")
    username = st.text_input("用户名（用户 ID，如 1001）", value="1001")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        resp = requests.post(
            f"{BACKEND_URL}/login",
            json={"username": username, "password": password},
        )
        if resp.status_code == 200:
            st.session_state["token"] = resp.json()["access_token"]
            st.success("登录成功")
            st.rerun()          # 重新运行脚本，进入聊天区
        else:
            st.error("用户名或密码错误")
    st.stop()                   # 未登录：停止渲染后面的聊天界面


# ========== 聊天区 ==========
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

retry = bool(st.session_state["pending_request"]) and st.button("继续上次未完成的回答")
prompt = st.chat_input()

if prompt or retry:
    request_id = st.session_state["pending_request"] if retry else str(uuid.uuid4())
    if prompt:
        st.session_state["pending_request"] = None
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    def chat_stream():
        """使用外层 prompt、会话 ID 和 Token 请求 /chat，产出供页面显示的文字。

        非200显示接口错误；SSE中的action加工具前缀、error加错误前缀，
        answer显示正文，遇DONE结束。此处不执行工具、不保存后端会话。
        """
        endpoint = (f"/conversations/{st.session_state['session_id']}/requests/{request_id}/resume" if retry else "/chat")
        body = None if retry else {"message": prompt, "session_id": st.session_state["session_id"], "request_id": request_id}
        with requests.post(
            BACKEND_URL + endpoint,
            json=body,
            headers={"Authorization": f"Bearer {st.session_state['token']}"},   # 带上登录 token
            stream=True,
            timeout=(5, 120),
        ) as resp:
            if resp.status_code != 200:
                yield f"请求失败（{resp.status_code}）：{resp.json().get('detail', '请稍后重试')}"
                return
            for line in resp.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]            # 去掉 "data: " 前缀
                if payload == b"[DONE]":       # 结束标记
                    break
                item = json.loads(payload)     # 解析出 {type, content}
                if item["type"] == "action":
                    yield f"[工具] {item['content']}\n\n"   # 动作：带标记前缀
                elif item["type"] == "error":
                    if item.get("resumable") and item.get("request_id"):
                        st.session_state["pending_request"] = item["request_id"]
                    yield f"\n\n[错误] {item['content']}"
                else:
                    yield item["content"]      # 回答：正常显示

    with st.chat_message("assistant"):
        if retry:
            st.session_state["pending_request"] = None
        try:
            reply = st.write_stream(chat_stream())
        except requests.RequestException:
            st.session_state["pending_request"] = request_id
            reply = "连接中断，可尝试继续上次回答；若请求尚在处理中，请稍后再试。"
            st.error(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    if st.session_state["pending_request"]:
        st.rerun()
