飞书多模态 RAG 机器人 (Feishu Multi-Modal RAG Agent)

1. 项目概述与目标

本项目是一个企业级的 多模态 RAG (检索增强生成) 概念验证Demo。旨在将基于本地复杂图文知识库的问答能力，接入飞书企业自建应用机器人。

2. 核心应用场景：

用户在飞书端发送自然语言问题 -> 机器人基于指定的 PDF 文档（包含纯文本、复杂表格、业务架构图等）进行精准检索 -> 输出结构化的高质量文本回答，并将底层依赖的原图精准推送给用户，实现图文并茂的 AI 智能助理体验。

3. 核心架构与处理流水线 (Pipeline)

拆分 (Partitioning): 使用 Unstructured 将 PDF 解构为原子级元素（文本块、表格、图像）;

智能分块 (Chunking): 采用 chunk_by_title ，用标题分割chunk;

多模态摘要 (LLM_summar): 调用 LLM 为复杂表格和图片生成增强型文本描述;

向量存储 (Vector Store）: 将chunk向量化存入 ChromaDB，将原始 Base64 图片等数据全量存储在 Metadata 中;

检索与生成 (Retrieval): 基于摘要进行语义召回，由多模态大模型 (Qwen-VL) 回答。

4. 核心技术栈 (Tech Stack)

编程语言: Python 3.12

应用框架: FastAPI (构建飞书 Webhook 服务), LangChain (RAG 核心编排)

解析PDF: unstructured[all-docs], poppler-utils, tesseract-ocr,libmagic-dev

向量数据库: ChromaDB 

模型服务 (阿里云 DashScope):

视觉大模型 (VLM): qwen3-vl-plus

向量嵌入模型 (Embedding): text-embedding-v3

接入平台：飞书开放平台 (Feishu Open Platform)

6. 项目目录结构
```text
feishu-multimodal-rag/
├── .env                    # 环境变量 (飞书密钥、API Key、Ngrok Token) [未开源]
├── .gitignore              # Git 忽略配置
├── requirements.txt        # 项目依赖清单
├── main.py                 # FastAPI Web服务主入口 (启动服务)
├── README.md               # 项目说明文档
├── doc/
│   └── source_document.pdf # 待处理的业务源 PDF 文件
├── data/                   # 结构化导出的中间态 JSON 数据目录
├── vector_db/              # ChromaDB 本地持久化存储库 [动态生成]
└── src/                    # RAG 核心逻辑库
    ├── ingestion_pipeline.py # PDF 处理全流程总指挥脚本
    ├── partition.py        # PDF 多模态元素提取与解析
    ├── chunk.py            # 动态分块策略
    ├── LLM_summar.py       # 大模型增强描述生成
    ├── vector_store.py     # 向量嵌入与入库逻辑
    ├── retrieval.py        # 原生 SDK 多模态检索与答案生成中枢
    └── utils.py            # 工具箱 (如导出 chunk 为 JSON 归档)
|__ .env.example            # 配置项目所需API
```

6. 环境准备

确保系统已安装 Python 3.12，建议使用 Conda 管理虚拟环境。

安装系统级依赖 (Ubuntu/WSL)：

```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
```

安装与运行

克隆仓库并安装 Python 依赖：

```bash
pip install -r requirements.txt
```
配置环境变量

复制.env.example为.env配置API_KEY

执行数据入库流水线：

```bash
python src/ingestion_pipeline.py
```
启动后端服务及内网穿透：

```bash
python main.py
```

7. 补充说明

飞书相关配置：FEISHU_APP_ID/APP_SECRET/VERIFICATION_TOKEN/ENCRYPT_KEY 均需在（飞书开放平台 - 自建应用 - 凭证与基础信息）中获取；

阿里云百炼：DASHSCOPE_API_KEY 在「阿里云百炼控制台 - API - 密钥管理」生成，QWEN_VL_MODEL 选择自己所需模型；

ngrok：NGROK_TOKEN 在 ngrok 官网（https://ngrok.com/）注册，在(Your Authtoken)中复制

check demo_video in the example folder
