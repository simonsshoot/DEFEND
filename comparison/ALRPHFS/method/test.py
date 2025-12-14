"""
临时测试脚本 - 用于调试 Bus Error
测试是否能正常加载模型和生成embedding
"""
import torch
from transformers import AutoTokenizer, AutoModel
import traceback

def test_model_loading():
    """测试模型加载"""
    model_path = "/data/Content_Moderation/BAAI-bge-m3"
    
    print("=" * 80)
    print("测试 1: 模型加载")
    print("=" * 80)
    
    try:
        print(f"正在加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        print("✓ 模型加载成功（CPU模式）")
        return tokenizer, model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return None, None

def test_single_embedding(tokenizer, model):
    """测试单个embedding生成"""
    print("\n" + "=" * 80)
    print("测试 2: 单个 Embedding 生成")
    print("=" * 80)
    
    test_text = "This is a test for embedding generation."
    
    try:
        print(f"测试文本: {test_text}")
        
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"Tokenization 完成，输入形状: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embedding = embeddings[0].cpu().numpy().tolist()
        
        print(f"✓ Embedding 生成成功，维度: {len(embedding)}")
        return True
    except Exception as e:
        print(f"✗ Embedding 生成失败: {e}")
        traceback.print_exc()
        return False

def test_batch_embeddings(tokenizer, model):
    """测试批量embedding生成"""
    print("\n" + "=" * 80)
    print("测试 3: 批量 Embedding 生成")
    print("=" * 80)
    
    test_texts = [
        "First test sentence for batch processing.",
        "Second test sentence with different content.",
        "Third test sentence to verify stability."
    ]
    
    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n  处理文本 {i}/{len(test_texts)}: {text[:50]}...")
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings[0].cpu().numpy().tolist()
            
            print(f"    ✓ 成功，维度: {len(embedding)}")
        
        print(f"\n✓ 批量处理完成，共处理 {len(test_texts)} 个文本")
        return True
    except Exception as e:
        print(f"\n✗ 批量处理失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始 Bus Error 调试测试...\n")
    
    # 测试1: 模型加载
    tokenizer, model = test_model_loading()
    
    if tokenizer is None or model is None:
        print("\n模型加载失败，无法继续测试")
        exit(1)
    
    # 测试2: 单个embedding
    if not test_single_embedding(tokenizer, model):
        print("\n单个embedding生成失败")
        exit(1)
    
    # 测试3: 批量embedding
    if not test_batch_embeddings(tokenizer, model):
        print("\n批量embedding生成失败")
        exit(1)
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！模型工作正常")
    print("✓ 可以继续运行 deduplicate.py")
    print("=" * 80)
