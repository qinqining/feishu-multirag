import json
from langchain_core.documents import Document
import dashscope
from dotenv import load_dotenv
load_dotenv()

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """使用 Qwen3-VL 创建多模态增强摘要"""
    
    try:
        
        # 1. 构建提示词文本
        instruction = """你的任务：
生成一份全面、便于检索的描述，需涵盖以下内容：
来自文本和表格的关键事实、数字与数据要点
所讨论的主要主题与核心概念
此内容能够回答的问题
视觉内容分析（图表、示意图、图片中的规律等）
用户可能使用的替代搜索词
请确保描述详细且便于检索 —— 优先考虑可查找性，而非简洁性。
可检索描述："""
        
        content_parts = [{"text": instruction}]
        
        # 2. 加入文本和表格素材
        prompt_body = f"\n【待分析文本内容】:\n{text}\n"
        if tables:
            prompt_body += "\n【表格数据】:\n"
            for i, table in enumerate(tables):
                prompt_body += f"表格 {i+1}:\n{table}\n"
        
        content_parts.append({"text": prompt_body})
        
        # 3. 加入图片素材 
        # images 应该是 base64 字符串或本地路径
        if images:
            for img in images:
                # 如果是本地路径，Qwen2-VL 接受 file:// 协议；如果是 base64，则按标准格式处理
                # 检查 img 是否是原始 Base64（即不包含 data: 前缀且不是 URL）
                if isinstance(img, str) and not img.startswith(('http', 'file://', 'data:')):
                    # 拼接标准的 Data URI 前缀
                    img_formatted = f"data:image/png;base64,{img}"
                else:
                    img_formatted = img
                
                content_parts.append({"image": img_formatted})

        # 4. 调用 DashScope (假设你已配置好环境变量)
        # 这里展示标准 SDK 调用，如果你用的是 LangChain 的 ChatDashScope 逻辑也类似
        response = dashscope.MultiModalConversation.call(
            model='qwen3-vl-plus-2025-12-19', 
            messages=[{
                'role': 'user',
                'content': content_parts
            }]
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content[0]['text']
        else:
            return f"Error: {response.code} - {response.message}"

    except Exception as e:
        return f"AI 摘要生成失败: {str(e)}"

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data