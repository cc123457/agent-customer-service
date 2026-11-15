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

prompt = st.chat_input()

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    def chat_stream():
        """流式调用后端 /chat，逐段产出回答"""
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": prompt, "session_id": st.session_state["session_id"]},
            headers={"Authorization": f"Bearer {st.session_state['token']}"},   # 带上登录 token
            stream=True,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]            # 去掉 "data: " 前缀
                if payload == b"[DONE]":       # 结束标记
                    break
                item = json.loads(payload)     # 解析出 {type, content}
                if item["type"] == "action":
                    yield f"[工具] {item['content']}\n\n"   # 动作：带标记前缀
                else:
                    yield item["content"]      # 回答：正常显示

    with st.chat_message("assistant"):
        reply = st.write_stream(chat_stream())
    st.session_state["messages"].append({"role": "assistant", "content": reply})
