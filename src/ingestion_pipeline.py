import os
import sys

# 把当前目录加入 Python 路径，防止找不到模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 严禁 Python 在本项目中生成缓存文件
sys.dont_write_bytecode = True

# 导入你拆分的四个模块
from partition import partition_document
from chunk import create_chunks_by_title
from LLM_summar import summarise_chunks
from vector_store import create_vector_store
from utils import export_chunks_to_json
def run_ingestion(pdf_path, db_path="vector_db/chroma_db"):
    """
    一键执行完整的数据入库流水线：拆分 -> 分块 -> 总结 -> 入库
    """
    print("\n Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    # --- Step 1: Partition ---
    print(f"\n[1/4] Partitioning Document: {pdf_path}...")
    elements = partition_document(pdf_path)
    print(f"✅ Extracted {len(elements)} elements.")

    # --- Step 2: Chunk ---
    print(f"\n[2/4] Chunking Elements...")
    chunks = create_chunks_by_title(elements)
    print(f"✅ Created {len(chunks)} chunks.")

    # --- Step 3: AI Summarisation ---
    print(f"\n[3/4] Generating AI Summaries (This may take a while)...")
    summarised_chunks = summarise_chunks(chunks)
    print(f"✅ Summarised {len(summarised_chunks)} chunks.")

    # +++ 新增的步骤：导出为 JSON 存档 +++
    print(f"\n[3.5/4] Exporting to JSON for inspection...")
    # 建议把 json 保存在 data 目录下
    json_path = os.path.join(project_root, "data", "summarised_chunks.json")
    export_chunks_to_json(summarised_chunks, filename=json_path)
    
    # --- Step 4: Vector Store ---
    print(f"\n[4/4] Creating Vector Store at: {db_path}...")
    db = create_vector_store(summarised_chunks, persist_directory=db_path)
    print(f"✅ Vector Store successfully created!")

    print("\n🎉 Pipeline completed successfully!")
    return db

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # 回退到 feishu-rag-demo/
    pdf_path = os.path.join(project_root, "doc", "视觉全流程指南.pdf")
    print(f"检查文件路径: {pdf_path}") 
    # 执行流水线
    run_ingestion(pdf_path)