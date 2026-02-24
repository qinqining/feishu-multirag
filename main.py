import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple

import uvicorn
import aiohttp
from aiohttp import FormData
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, Request
from Crypto.Cipher import AES
import base64
import hashlib
import uuid

# --- 1. 环境与基础配置 ---
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

# 把项目根目录加入系统路径，方便导入 src 下的模块
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 从环境变量获取飞书配置
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET")
FEISHU_ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY")

# 全局变量：用于幂等去重（防止飞书重试导致重复回复）
processed_messages = set()

# ⚠️ 导入你的 RAG 检索模块 (根据你的新 Pipeline，这里应该替换为真实的检索函数)
# 我们假设你在 src/retrieval.py 中写了一个 get_answer 函数
try:
    from src.retrieval import get_answer
except ImportError:
    logger.warning("⚠️ 未找到 src.retrieval.get_answer，将使用模拟回答测试飞书链路。")
    def get_answer(query: str) -> Tuple[str, List[str]]:
        """模拟的检索函数，返回: (文本答案, [本地图片路径列表])"""
        return f"这是关于『{query}』的测试回答。", []

# --- 2. 飞书 AES 解密类 ---
class AESCipher:
    def __init__(self, key):
        self.key = hashlib.sha256(key.encode('utf-8')).digest()

    def decrypt(self, encrypt_text):
        encrypt_text = base64.b64decode(encrypt_text)
        cipher = AES.new(self.key, AES.MODE_CBC, encrypt_text[:16])
        slice_decrypted = cipher.decrypt(encrypt_text[16:])
        padding_count = slice_decrypted[-1]
        decrypted_text = slice_decrypted[:-padding_count]
        return decrypted_text.decode('utf-8')

# --- 3. 飞书 API 交互工具 ---
async def get_feishu_token() -> str:
    """获取飞书 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = {"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                res = await response.json()
                return res.get("tenant_access_token", "") if res.get("code") == 0 else ""
    except Exception as e:
        logger.error(f"❌ 获取 Token 失败: {e}")
        return ""

async def upload_base64_image_to_feishu(base64_data: str) -> str:
    """直接将 Base64 字符串在内存中转换并上传到飞书，返回 image_key"""
    token = await get_feishu_token()
    if not token or not base64_data:
        return ""

    # 1. 自动清理 Base64 字符串 (防呆设计：去掉可能存在的 data:image/jpeg;base64, 前缀)
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]

    # 2. 在内存中解码成图片 Bytes
    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        logger.error(f"❌ Base64 解码失败: {e}")
        return ""
    
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    try:
        form = FormData()
        form.add_field('image_type', 'message')
        
        # 飞书接口强制要求提供一个 filename，我们用 uuid 随机捏造一个给他
        random_filename = f"rag_image_{uuid.uuid4().hex[:8]}.jpg"
        form.add_field('image', image_bytes, filename=random_filename)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Authorization": f"Bearer {token}"}, data=form) as response:
                res = await response.json()
                if res.get("code") == 0:
                    key = res.get("data", {}).get("image_key", "")
                    logger.info(f"✅ Base64 图片上传飞书成功 -> {key}")
                    return key
                else:
                    logger.error(f"❌ 飞书接口返回错误: {res}")
                    return ""
    except Exception as e:
        logger.error(f"❌ 上传 Base64 图片至飞书崩溃: {e}")
        return ""

def build_feishu_card(answer: str, question: str, image_keys: List[str]) -> Dict:
    """构建飞书富文本消息卡片"""
    elements = [
        {"tag": "div", "text": {"tag": "lark_md", "content": f"**🙋 问：{question}**"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": f"**🤖 答：**\n{answer}"}}
    ]
    
    if image_keys:
        elements.append({"tag": "hr"}) # 添加分割线
        for key in image_keys:
            elements.append({
                "tag": "img",
                "img_key": key,
                "mode": "fit_horizontal",
                "alt": {"tag": "plain_text", "content": "相关插图"}
            })
            
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": "📄 视觉全流程助手"}, "template": "blue"},
        "elements": elements
    }

# --- 4. 核心业务：后台 RAG 处理逻辑 ---
async def handle_rag_logic(msg_id: str, question: str):
    """专门处理 RAG 和回复的异步后台任务"""
    try:
        logger.info(f"🧠 开始处理问题: {question}")
        
        # 1. 调用你新写的 RAG Pipeline 检索答案
        # 【注意】这里 get_answer 返回的第二个参数，变成了 Base64 字符串列表！
        answer_text, image_base64_list = get_answer(question)
        
        # 2. 处理图片：如果有图片，直接在内存中解码并上传
        final_image_keys = []
        for b64_str in image_base64_list[:3]:
            logger.info("🚀 正在并发上传图片至飞书...")
            upload_tasks = [upload_base64_image_to_feishu(b64) for b64 in image_base64_list[:3]]
            keys = await asyncio.gather(*upload_tasks)
            final_image_keys = [k for k in keys if k]

        # 3. 构建消息卡片 (这步不需要改)
        card_content = build_feishu_card(answer_text, question, final_image_keys)
        
        # 4. 回复用户
        token = await get_feishu_token()
        reply_url = f"https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                reply_url, 
                headers={"Authorization": f"Bearer {token}"}, 
                json={"content": json.dumps(card_content), "msg_type": "interactive"}
            ) as resp:
                send_res = await resp.json()
                if send_res.get('code') == 0:
                    logger.info(f"📩 飞书卡片回复成功! MsgID: {msg_id}")
                else:
                    logger.error(f"❌ 飞书卡片回复失败: {send_res}")
                    
    except Exception as e:
        logger.error(f"❌ 异步处理任务崩溃: {e}", exc_info=True)


# --- 5. FastAPI 路由入口 ---
app = FastAPI()

@app.post("/api/feishu/webhook")
async def feishu_webhook(request: Request):
    """飞书事件订阅统一入口"""
    body = await request.body()
    data = json.loads(body.decode("utf-8"))
    
    # 1. 解密逻辑 (如果配置了 Encrypt Key)
    if "encrypt" in data:
        if not FEISHU_ENCRYPT_KEY:
            logger.error("❌ 收到加密消息，但未配置 FEISHU_ENCRYPT_KEY")
            return {"ok": False}
        try:
            cipher = AESCipher(FEISHU_ENCRYPT_KEY)
            decrypted_json = cipher.decrypt(data["encrypt"])
            data = json.loads(decrypted_json)
        except Exception as e:
            logger.error(f"❌ 数据解密失败: {e}")
            return {"ok": False}

    # 2. 飞书 URL 验证 (配置 Webhook 时的第一次握手)
    if data.get("type") == "url_verification":
        logger.info("✅ 收到飞书 URL 验证请求")
        return {"challenge": data.get("challenge")}

    # 3. 解析事件内容
    header = data.get("header", {})
    event = data.get("event", {})
    event_type = header.get("event_type")
    
    # 4. 处理接收消息事件
    if event_type == "im.message.receive_v1":
        msg = event.get("message", {})
        msg_id = msg.get("message_id")
        
        # 【重要】幂等处理：防止飞书因超时重试导致机器人重复发消息
        if msg_id in processed_messages:
            logger.warning(f"⚠️ 收到重复消息，已忽略: {msg_id}")
            return {"ok": True}
        processed_messages.add(msg_id)
        
        # 提取用户发送的纯文本内容 (去掉 @ 机器人的部分)
        content_json = json.loads(msg.get("content", "{}"))
        import re
        question = re.sub(r"@[^ ]+ ", "", content_json.get("text", "")).strip()
        
        # 【重要】立即启动后台任务，然后马上 return 给飞书 200 OK
        asyncio.create_task(handle_rag_logic(msg_id, question))
        
        return {"ok": True}

    # 其他未处理的事件也返回 OK，防止飞书一直重发
    return {"ok": True}


# --- 6. 启动程序 ---
if __name__ == "__main__":
    # 启动 ngrok 内网穿透 (如果你在本地测试)
    from pyngrok import ngrok
    import os
    token = os.getenv("NGROK_TOKEN")
    if token:
        ngrok.set_auth_token(token)
        try:
            public_url = ngrok.connect(8000).public_url
            logger.info("="*50)
            logger.info(f"🌍 飞书 Webhook 地址: {public_url}/api/feishu/webhook")
            logger.info("👉 请将上方地址复制到飞书开放平台 -> 事件订阅 -> 请求地址 中")
            logger.info("="*50)
        except Exception as e:
            logger.error(f"⚠️ Ngrok 启动失败: {e}")

    # 启动 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)