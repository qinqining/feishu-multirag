import os
import json
from typing import Tuple, List
from dotenv import load_dotenv
from loguru import logger

from langchain_community.vectorstores import Chroma
from langchain_dashscope import DashScopeEmbeddings
import dashscope
from dashscope import MultiModalConversation


load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "vector_db", "chroma_db")


embeddings = DashScopeEmbeddings(model="text-embedding-v3")
try:
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    logger.info("✅ 成功连接 ChromaDB 向量库！")
except Exception as e:
    logger.error(f"❌ 连接 ChromaDB 失败: {e}")
    vector_store = None


def get_answer(query: str) -> Tuple[str, List[str]]:
    """使用阿里原生 MultiModalConversation 接口生成回答"""
    if not vector_store:
        return "抱歉，向量数据库未初始化。", []

    logger.info(f"🔍 正在检索问题: {query}")
    
    chunks = vector_store.similarity_search(query, k=2)
    
    if not chunks:
        return "抱歉，知识库中未找到相关内容。", []

    try:
        prompt_text = f"请使用上述文本、表格和图片，提供清晰、全面的答案。如果文档中没有足够的信息来回答该问题，请说明：“根据提供的文档，我没有足够的信息来回答这个问题{query}\n\n内容：\n"
        message_content = []
        all_images_base64 = [] # 用于交给 main.py 上传飞书
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- 分块 {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                try:
                    meta = json.loads(chunk.metadata["original_content"])
                    prompt_text += f"文字内容：\n{meta.get('raw_text', '')}\n"
                    
                    # 处理图片：组装成原生 SDK 要求的格式
                    for img_b64 in meta.get("images_base64", []):
                        # 清洗确保有正确前缀
                        clean_b64 = img_b64.split(",")[-1] if "," in img_b64 else img_b64
                        fixed_img = f"data:image/jpeg;base64,{clean_b64}"
                        
                        # 加入模型上下文
                        message_content.append({"image": fixed_img})
                        # 存入列表交回给飞书
                        all_images_base64.append(clean_b64)
                except json.JSONDecodeError:
                    prompt_text += f"{chunk.page_content}\n"
            else:
                prompt_text += f"{chunk.page_content}\n"

        # 将 Prompt 文本插入到消息数组的首位 (和你的前端逻辑一模一样)
        message_content.insert(0, {"text": prompt_text})

        logger.info("🧠 正在通过阿里原生多模态 SDK 呼叫 Qwen3-VL-Plus...")
        
        # 🚨 核心改动：使用能 100% 跑通的原生调用方式
        response = MultiModalConversation.call(
            model='qwen3-vl-plus',  # 或者填你前端用的具体版本号
            messages=[{"role": "user", "content": message_content}]
        )

        # 解析原生 SDK 的返回结果
        if response.status_code == 200:
            # 拿到最终的文字回答
            answer = response.output.choices[0].message.content[0]['text']
            logger.info("✅ 原生接口调用成功，回答已生成！")
            return answer, all_images_base64
        else:
            logger.error(f"❌ 阿里云接口报错: {response.code} - {response.message}")
            return f"抱歉，大模型分析失败：{response.message}", []
            
    except Exception as e:
        logger.error(f"❌ 回答生成过程发生代码异常: {e}")
        return "抱歉，系统处理时发生内部故障。", []