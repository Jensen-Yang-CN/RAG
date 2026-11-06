"""将法条块向量写入 Milvus Lite，并评测 LeCoQA 测试集。"""

import json
import time
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).parent
OUTPUT = ROOT / "data" / "processed"
CHUNKS = OUTPUT / "chunks.jsonl"
TEST = ROOT / "LeCoQA" / "data" / "example" / "test.json"
DB = OUTPUT / "legal_vectors.db"
PROGRESS = OUTPUT / "vector_import_progress.txt"
COLLECTION = "legal_chunks_bge_small_zh_v15"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
BATCH_SIZE = 128


def build_index(client, model, chunks):
    if not client.has_collection(COLLECTION):
        client.create_collection(collection_name=COLLECTION, dimension=512, metric_type="COSINE")
        PROGRESS.write_text("0", encoding="utf-8")

    start = int(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else 0
    for offset in range(start, len(chunks), BATCH_SIZE):
        batch = chunks[offset:offset + BATCH_SIZE]
        texts = [row["name"] + " " + row["content"] for row in batch]
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
        data = [
            {
                "id": offset + i,
                "vector": vectors[i].tolist(),
                "chunk_id": row["chunk_id"],
                "article_id": row["article_id"],
                "name": row["name"],
                "content": row["content"],
            }
            for i, row in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION, data=data)
        PROGRESS.write_text(str(offset + len(batch)), encoding="utf-8")
        print(f"已导入 {offset + len(batch)}/{len(chunks)} 块", flush=True)


def load_index(client):
    if client.describe_index(COLLECTION, "vector")["index_type"] != "FLAT":
        client.drop_index(collection_name=COLLECTION, index_name="vector")
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(collection_name=COLLECTION, index_params=index)
    client.load_collection(COLLECTION)


def search(client, model, question, top_k=10):
    vector = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    limit = max(100, top_k * 2)
    while True:
        hits = client.search(
            collection_name=COLLECTION,
            data=[vector],
            limit=limit,
            output_fields=["article_id", "name"],
        )[0]
        articles = {}
        for hit in hits:
            article_id = hit["entity"]["article_id"]
            if article_id not in articles or hit["distance"] > articles[article_id]["score"]:
                articles[article_id] = {
                    "article_id": article_id,
                    "name": hit["entity"]["name"],
                    "score": float(hit["distance"]),
                }
        if len(articles) >= top_k or len(hits) < limit:
            return sorted(articles.values(), key=lambda row: (-row["score"], row["article_id"]))[:top_k]
        limit *= 2


def evaluate(client, model, chunks):
    tests = json.load(TEST.open(encoding="utf-8"))
    recall5 = recall10 = mrr10 = total_ms = 0.0
    for test in tests:
        start = time.perf_counter()
        top10 = search(client, model, test["问题"])
        top_ids = [row["article_id"] for row in top10]
        total_ms += (time.perf_counter() - start) * 1000

        gold = set(test["match_id"])
        recall5 += len(gold.intersection(top_ids[:5])) / len(gold)
        recall10 += len(gold.intersection(top_ids)) / len(gold)
        for rank, article_id in enumerate(top_ids, start=1):
            if article_id in gold:
                mrr10 += 1 / rank
                break

    count = len(tests)
    metrics = {
        "model": MODEL_NAME,
        "test_queries": count,
        "indexed_chunks": len(chunks),
        "Recall@5": round(recall5 / count, 4),
        "Recall@10": round(recall10 / count, 4),
        "MRR@10": round(mrr10 / count, 4),
        "mean_query_ms": round(total_ms / count, 2),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    model = SentenceTransformer(MODEL_NAME)
    client = MilvusClient(str(DB))
    build_index(client, model, chunks)
    load_index(client)
    evaluate(client, model, chunks)
    client.close()


if __name__ == "__main__":
    main()
