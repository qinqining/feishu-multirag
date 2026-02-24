import json
import os

def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format for inspection"""
    print(f" 正在导出 Chunks 到 JSON 文件...")
    export_data = []
    
    for i, doc in enumerate(chunks):
        # 尝试安全地解析 original_content
        original_content = doc.metadata.get("original_content", "{}")
        if isinstance(original_content, str):
            try:
                original_content = json.loads(original_content)
            except json.JSONDecodeError:
                pass # 如果解析失败，就保留原样
                
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content, # 这里通常是 LLM 生成的总结
            "metadata": {
                "original_content": original_content,
                # 如果有其他 metadata 比如 page_number，也可以在这里加上
                "category": doc.metadata.get("category", "unknown")
            }
        }
        export_data.append(chunk_data)
    
    # 确保保存的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save to file (ensure_ascii=False 保证中文正常显示，不变成 \uXXXX)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 成功导出 {len(export_data)} 个 chunks 到: {filename}")
    return export_data