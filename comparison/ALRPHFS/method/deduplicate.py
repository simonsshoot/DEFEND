import numpy as np
from typing import Dict, List
from sklearn.cluster import DBSCAN
import chromadb
import hashlib, json
from chromadb.config import Settings
from typing import Dict, List
import spacy
from rank_bm25 import BM25Okapi
nlp = spacy.load("en_core_web_sm")
import pickle
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel

# 全局变量，用于缓存模型
_embedding_model = None
_embedding_tokenizer = None

def load_local_embedding_model(model_path="/data/Content_Moderation/BAAI-bge-m3"):
    global _embedding_model, _embedding_tokenizer
    
    if _embedding_model is None:
        _embedding_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _embedding_model = AutoModel.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            _embedding_model = _embedding_model.cuda()
        _embedding_model.eval()
        print("模型加载完成")
    return _embedding_tokenizer, _embedding_model


def get_embedding(text, model_path="/data/Content_Moderation/BAAI-bge-m3"):
    """
    使用本地 BGE-M3 模型获取文本的 embedding 向量。
    :param text: 要转换的文本 (str)
    :param model_path: 本地模型路径
    :return: embedding向量 (list)
    """
    try:
        tokenizer, model = load_local_embedding_model(model_path)

        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 [CLS] token 的 embedding 或者 mean pooling
            # 这里使用 mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embedding = embeddings[0].cpu().numpy().tolist()
        
        return embedding
    
    except Exception as e:
        print(f"获取 embedding 时出错: {e}")
        return None
def cluster_items_dbscan_group_by_label(
    items: List[Dict],
    field: str = "attack_essence",
    eps: float = 0.3,
    min_samples: int = 1
) -> Dict[int, Dict[str, List]]:
    """
    使用 DBSCAN 聚类，按 label 分组返回 items 和 embeddings。
    返回结构: {label: {"items": [...], "embeddings": [...]}}
    """
    embeddings = []
    valid_items = []

    for item in items:
        text = item[field]
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(np.array(emb))
            valid_items.append(item)

    if not embeddings:
        return {}

    embeddings_array = np.vstack(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(embeddings_array)

    label_groups = {}

    for idx, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = {"items": [], "embeddings": []}
        label_groups[label]["items"].append(valid_items[idx])
        label_groups[label]["embeddings"].append(embeddings[idx].tolist())

    return label_groups
# import Optional
def greedy_multi_medoid_selection(
    embeddings: List[List[float]],
    top_k: int = None,
    min_required_dist: float = 0.2
) -> List[int]:
    """
    从一个簇内的嵌入中贪心选择最多 top_k 个 medoids，
    要求每个新选择的点与已有 medoids 之间的最小距离 >= min_required_dist。
    """
    if not embeddings:
        return []
    top_k = len(embeddings) // 2
    embs = np.array(embeddings)
    num_points = embs.shape[0]
    if num_points == 1:
        return [0]

    # 计算余弦距离矩阵 (1 - cosine similarity)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized_embs = embs / (norms + 1e-10)
    sim_matrix = np.dot(normalized_embs, normalized_embs.T)
    dist_matrix = 1 - sim_matrix

    # Step 1: 找到第一个 medoid（离所有点平均距离最近的点）
    avg_dists = dist_matrix.mean(axis=1)
    first_medoid = int(np.argmin(avg_dists))
    medoids = [first_medoid]

    # Step 2: 贪心选择后续 medoids
    while len(medoids) < top_k:
        max_min_dist = -1
        best_candidate = -1
        for i in range(num_points):
            if i in medoids:
                continue
            min_dist_to_medoids = min(dist_matrix[i][j] for j in medoids)
            if min_dist_to_medoids > max_min_dist:
                max_min_dist = min_dist_to_medoids
                best_candidate = i

        if max_min_dist >= min_required_dist:
            medoids.append(best_candidate)
        else:
            break

    return medoids


def select_greedy_medoids_from_label_groups(
    clustered_items: Dict[int, Dict[str, List]],
    top_k: int = 3,
    min_required_dist: float = 0.2
) -> Dict[int, Dict[str, List]]:
    """
    在每个簇内使用“贪心多‑Medoid”策略选出 top_k 个代表样本：
      1. 先选第一个 Medoid：簇中心（质心）最近的点。
      2. 之后每次选出一个，使它与已有 Medoid 集合的最小距离最大（farthest‑first）。
    :param clustered_items:
        { label: { "items": [...], "embeddings": [...] } }
    :param top_k: 每簇想要保留的 Medoid 数量
    :return: 同格式字典，只保留每簇的 top_k 个 Medoid
    """
    result: Dict[int, Dict[str, List]] = {}

    for label, group in clustered_items.items():
        items = group["items"]
        embs = np.array(group["embeddings"])  # shape = (n_points, dim)

        # 如果样本不超过 top_k，就全部保留
        if embs.shape[0] <= top_k:
            result[label] = {
                "items": items.copy(),
                "embeddings": embs.tolist()
            }
            continue

        # 1. 使用贪心多‑Medoid策略选出 top_k 个 medoids
        medoid_indices = greedy_multi_medoid_selection(embeddings=embs.tolist(), min_required_dist=min_required_dist)

        # 2. 收集结果
        selected_items = [items[i] for i in medoid_indices]
        selected_embs  = [embs[i].tolist() for i in medoid_indices]
        result[label] = {
            "items": selected_items,
            "embeddings": selected_embs
        }

    return result

def default_bm25_tokenizer(text: str):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

def save_medoid_groups_to_chroma_and_bm25(
        medoid_groups: Dict[int, Dict[str, List]],
        field: str = "attack_essence",
        persist_dir: str = r"C:\path\to\persist",
        collection_name: str = "medoid_collection",
        bm25_storage_path: str = "./bm25_data",
        save_bm25: bool = True  # 控制是否保存 BM25
):
    """
    将 medoid_groups 写入 ChromaDB，并可选地构建并保存 BM25 索引。
    Args:
        medoid_groups: 聚类结果字典
        field: 要存储的字段名
        persist_dir: ChromaDB 持久化目录
        collection_name: 集合名称
        bm25_storage_path: BM25 索引持久化目录
        save_bm25: 是否构建并保存 BM25 索引
    """
    # 1. 连接或创建 ChromaDB 集合
    client = chromadb.Client(Settings(persist_directory=persist_dir, anonymized_telemetry=False))
    try:
        col = client.get_collection(name=collection_name)
    except Exception:
        col = client.create_collection(name=collection_name, metadata={"hnsw": "cosine"})

    ids, docs, embs, metas = [], [], [], []
    # 准备 BM25 数据
    bm25_corpus = []
    doc_list = []

    # 2. 扁平化所有簇
    for label, group in medoid_groups.items():
        for item, emb in zip(group["items"], group["embeddings"]):
            base = item.get("attack_essence", "")
            _id = hashlib.md5(base.encode("utf-8")).hexdigest()
            doc = item[field]

            meta = {}
            for k, v in item.items():
                if isinstance(v, (str, bool, int, float)):
                    meta[k] = v
                else:
                    try:
                        meta[k] = json.dumps(v, ensure_ascii=False)
                    except:
                        meta[k] = str(v)
            meta["_cluster"] = int(label)

            ids.append(_id)
            docs.append(doc)
            embs.append(emb)
            metas.append(meta)

            # BM25 数据
            if save_bm25:
                doc_list.append(doc)
                tokens = default_bm25_tokenizer(doc)
                bm25_corpus.append(tokens)

    # 3. 批量写入 ChromaDB
    col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    print(f"已写入 {len(ids)} 个样本到 ChromaDB 集合 '{collection_name}'。")

    # 4. 可选：构建并保存 BM25
    if save_bm25 and bm25_corpus:
        bm25_index = BM25Okapi(bm25_corpus)
        os.makedirs(bm25_storage_path, exist_ok=True)
        # 保存 id_list, doc_list 和 bm25_corpus
        bm25_data = {
            "id_list": ids,
            "doc_list": doc_list,
            "bm25_corpus": bm25_corpus
        }
        with open(os.path.join(bm25_storage_path, "bm25_data.pkl"), "wb") as f:
            import pickle;
            pickle.dump(bm25_data, f)
        print(f"已创建并保存 BM25 索引，包含 {len(bm25_corpus)} 个文档。")


def save_medoids_to_json(medoid_groups: Dict[int, Dict[str, List]], output_path: str = "./deduplicated_agentharm.json"):
    """
    将去重后的 medoid 结果保存为 JSON 文件。
    
    :param medoid_groups: 去重后的结果字典
    :param output_path: 输出文件路径
    """
    output_data = []
    
    for label, group in medoid_groups.items():
        for item in group["items"]:
            item_copy = item.copy()
            item_copy["cluster_label"] = int(label)
            output_data.append(item_copy)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(output_data)} 条去重后的数据到: {output_path}")


def load_agentharm_risk_patterns(folder_path="./risk_patterns_agentharm"):
    """
    加载 AgentHarm 数据集的所有风险模式文件。
    遍历 folder_path 下所有 .json 文件，提取包含 attack_essence 的数据项。
    
    :param folder_path: AgentHarm 风险模式文件所在目录
    :return: 所有风险模式的列表
    """
    attacks = []
    
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return attacks
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"  - 读取文件: {filename}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    count = 0
                    for item in data:
                        if "attack_essence" in item:
                            attacks.append(item)
                            count += 1
            except Exception as e:
                print(f"    错误：读取 {filename} 时出错: {e}")
    print(f"\n总共加载了 {len(attacks)} 条风险模式数据\n")
    return attacks


if __name__ == "__main__":
    AGENTHARM_PATTERNS_DIR = "./risk_patterns_agentharm"
    EPS = 0.3            
    MIN_SAMPLES = 1        

    TOP_K_MEDOIDS = 3      # 每簇最多保留的代表性样本数
    MIN_REQUIRED_DIST = 0.2  # Medoid 之间的最小距离

    CHROMA_PERSIST_DIR = "./chroma_agentharm"
    BM25_STORAGE_PATH = "./bm25_agentharm"
    
    EMBEDDING_MODEL_PATH = "/data/Content_Moderation/BAAI-bge-m3"
    
    print("\n[Step 1] 加载 AgentHarm 风险模式数据...")
    attacks = load_agentharm_risk_patterns(folder_path=AGENTHARM_PATTERNS_DIR)
    
    if not attacks:
        print("错误：未加载到任何数据，请检查数据路径！")
        exit(1)
    
    print(f"\n[Step 2] 使用 DBSCAN 进行聚类（eps={EPS}, min_samples={MIN_SAMPLES}）...")
    clustered_result = cluster_items_dbscan_group_by_label(
        items=attacks,
        field="attack_essence",
        eps=EPS,
        min_samples=MIN_SAMPLES
    )

    print(f"\n[Step 3] 从每个簇中选择代表性样本（top_k={TOP_K_MEDOIDS}, min_dist={MIN_REQUIRED_DIST}）...")
    topk_medoid_groups = select_greedy_medoids_from_label_groups(
        clustered_items=clustered_result,
        top_k=TOP_K_MEDOIDS,
        min_required_dist=MIN_REQUIRED_DIST
    )
    
    total_medoids = sum(len(grp['items']) for grp in topk_medoid_groups.values())
    print(f"\n去重完成！从 {len(attacks)} 条原始数据中选出 {total_medoids} 条代表性样本")

    json_output_path = "./deduplicated_agentharm_patterns.json"
    save_medoids_to_json(topk_medoid_groups, output_path=json_output_path)

    save_medoid_groups_to_chroma_and_bm25(
        medoid_groups=topk_medoid_groups,
        field="attack_essence",
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name="attack_essence_agentharm",
        bm25_storage_path=os.path.join(BM25_STORAGE_PATH, "attack_essence"),
        save_bm25=True
    )
    
    save_medoid_groups_to_chroma_and_bm25(
        medoid_groups=topk_medoid_groups,
        field="harmful_result",
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name="harmful_result_agentharm",
        bm25_storage_path=os.path.join(BM25_STORAGE_PATH, "harmful_result"),
        save_bm25=False
    )
    
    print("\n" + "=" * 80)
    print(f"✓ 去重结果 JSON 文件: {json_output_path}")
    print(f"✓ ChromaDB 索引保存至: {CHROMA_PERSIST_DIR}")
    print(f"✓ BM25 索引保存至: {BM25_STORAGE_PATH}")
    print(f"✓ 原始数据量: {len(attacks)} 条")
    print(f"✓ 去重后数据量: {total_medoids} 条")
    print(f"✓ 去重比例: {total_medoids/len(attacks)*100:.2f}%")
    print("=" * 80)

    OUTPUT_JSON_PATH = "./deduplicated_agentharm.json"
    save_medoids_to_json(topk_medoid_groups, output_path=OUTPUT_JSON_PATH)
