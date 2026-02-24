# for linux
# !apt-get install poppler-utils(处理 PDF 文件（提取文本、转换格式等) tesseract-ocr(OCR 文字识别工具) libmagic-dev(文件类型检测库)
#sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
load_dotenv()

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f" Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,  
        strategy="hi_res", #设定PDF解析的核心策略
        infer_table_structure=True, # 保留表格的结构化格式, not jumbled text
        extract_image_block_types=["Image"], #  指定要提取的图片类型
        extract_image_block_to_payload=True, # 将图片转换为可使用的base64格式存储
        languages=[ "chi_sim"] , #  指定OCR识别的语言（简体中文）   
    )

    images = [el for el in elements if el.category == 'Image']
    tables = [el for el in elements if el.category == 'Table']
        
    print(f"Partitioning Complete!")
    print(f"Statistics:")
    print(f"   - Total Elements: {len(elements)}")
    print(f"   - Images Found:   {len(images)}")
    print(f"   - Tables Found:   {len(tables)}")
    if len(images) == 0:
        print("⚠️ Warning: No images found. Check if 'poppler' and 'tesseract' are installed correctly.")
        print("🚀 准备发货！当前的 elements 数量是:", len(elements))
    return elements