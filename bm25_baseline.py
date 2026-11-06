"""用 LeCoQA 测试集评估法条的 BM25 检索效果。"""

import json
import time
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
TEST = ROOT / "LeCoQA" / "data" / "example" / "test.json"
OUTPUT = ROOT / "data" / "processed"


def tokenize(text):
    return [word for word in jieba.lcut(text) if word.isalnum()]


def search(index, article_ids, article_names, question, top_k=10):
    scores = index.get_scores(tokenize(question))
    article_scores = {}
    for article_id, score in zip(article_ids, scores):
        if article_id not in article_scores or score > article_scores[article_id]:
            article_scores[article_id] = float(score)
    top_ids = sorted(article_scores, key=lambda i: (-article_scores[i], i))[:top_k]
    return [
        {"article_id": i, "name": article_names[i], "score": article_scores[i]}
        for i in top_ids
    ]


def main():
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    tests = json.load(TEST.open(encoding="utf-8"))
    article_ids = [chunk["article_id"] for chunk in chunks]
    article_names = {chunk["article_id"]: chunk["name"] for chunk in chunks}

    index = BM25Okapi([tokenize(chunk["name"] + " " + chunk["content"]) for chunk in chunks])

    recall5 = 0.0
    recall10 = 0.0
    mrr10 = 0.0
    total_ms = 0.0
    for test in tests:
        start = time.perf_counter()
        top10 = search(index, article_ids, article_names, test["问题"])
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
        "test_queries": count,
        "indexed_chunks": len(chunks),
        "Recall@5": round(recall5 / count, 4),
        "Recall@10": round(recall10 / count, 4),
        "MRR@10": round(mrr10 / count, 4),
        "mean_query_ms": round(total_ms / count, 2),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
