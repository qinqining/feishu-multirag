from langchain_dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import time

def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """分批创建并持久化 ChromaDB 向量库"""
    print(f"🔮 开始处理 {len(documents)} 个文档，采用分批处理模式...")
    
    embedding_model = DashScopeEmbeddings(model="text-embedding-v3")
    
    batch_size = 10  # 按照报错提示，限制为 10 条一组
    vectorstore = None

    # 将文档列表拆分成 10 个一组
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"--- 正在处理第 {current_batch_num}/{total_batches} 批次 ({len(batch)} 条数据) ---")
        
        try:
            if vectorstore is None:
                # 第一批次：创建并初始化向量库
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                # 后续批次：向已有的向量库添加文档
                vectorstore.add_documents(documents=batch)
            
            # 适当留出一点点冷却时间，防止触发 API 频率限制（QPS）
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"❌ 第 {current_batch_num} 批次处理失败: {e}")
            # 这里可以选择 continue 跳过，或者 raise 报错
            continue

    print(f"✅ 所有批次处理完成，向量库已保存至 {persist_directory}")
    return vectorstore