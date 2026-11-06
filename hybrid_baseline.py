"""融合 BM25 和向量检索的前 100 条法条，评测 RRF 混合检索。"""

import json
import time

from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import bm25_baseline as bm25
import vector_baseline as dense


CANDIDATES = 100
RRF_K = 60


def fuse(bm25_hits, vector_hits):
    articles = {}
    for method, hits in [("bm25", bm25_hits), ("vector", vector_hits)]:
        for rank, hit in enumerate(hits, start=1):
            article_id = hit["article_id"]
            if article_id not in articles:
                articles[article_id] = {
                    "article_id": article_id,
                    "name": hit["name"],
                    "score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                }
            articles[article_id]["score"] += 1 / (RRF_K + rank)
            articles[article_id][method + "_rank"] = rank
    return sorted(articles.values(), key=lambda row: (-row["score"], row["article_id"]))


def calculate_metrics(method, rows, chunk_count):
    recall5 = recall10 = mrr10 = 0.0
    for row in rows:
        gold = set(row["gold_article_ids"])
        top_ids = [hit["article_id"] for hit in row["top10"]]
        recall5 += len(gold.intersection(top_ids[:5])) / len(gold)
        recall10 += len(gold.intersection(top_ids)) / len(gold)
        for rank, article_id in enumerate(top_ids, start=1):
            if article_id in gold:
                mrr10 += 1 / rank
                break

    count = len(rows)
    metrics = {
        "test_queries": count,
        "indexed_chunks": chunk_count,
        "candidate_articles_per_route": CANDIDATES,
        "Recall@5": round(recall5 / count, 4),
        "Recall@10": round(recall10 / count, 4),
        "MRR@10": round(mrr10 / count, 4),
        "mean_query_ms": round(sum(row["query_ms"] for row in rows) / count, 2),
    }
    if method in ("vector", "hybrid"):
        metrics["model"] = dense.MODEL_NAME
    if method == "hybrid":
        metrics["rrf_k"] = RRF_K
    return metrics


def main():
    chunks = [json.loads(line) for line in bm25.CHUNKS.open(encoding="utf-8")]
    tests = json.load(bm25.TEST.open(encoding="utf-8"))
    article_ids = [row["article_id"] for row in chunks]
    article_names = {row["article_id"]: row["name"] for row in chunks}
    index = BM25Okapi([bm25.tokenize(row["name"] + " " + row["content"]) for row in chunks])
    model = SentenceTransformer(dense.MODEL_NAME)
    client = MilvusClient(str(dense.DB))
    dense.load_index(client)
    results = {method: [] for method in ("bm25", "vector", "hybrid")}

    try:
        for number, test in enumerate(tests, start=1):
            start = time.perf_counter()
            bm25_hits = bm25.search(index, article_ids, article_names, test["问题"], CANDIDATES)
            bm25_end = time.perf_counter()
            vector_hits = dense.search(client, model, test["问题"], CANDIDATES)
            vector_end = time.perf_counter()
            hybrid_hits = fuse(bm25_hits, vector_hits)
            end = time.perf_counter()

            for method, hits, seconds in [
                ("bm25", bm25_hits, bm25_end - start),
                ("vector", vector_hits, vector_end - bm25_end),
                ("hybrid", hybrid_hits, end - start),
            ]:
                results[method].append({
                    "query_id": test["query_id"],
                    "gold_article_ids": test["match_id"],
                    "top10": hits[:10],
                    "query_ms": round(seconds * 1000, 2),
                })
            if number % 25 == 0 or number == len(tests):
                print(f"Evaluated {number}/{len(tests)}", flush=True)
    finally:
        client.close()

    print("Method | Recall@5 | Recall@10 | MRR@10 | Mean query ms")
    for method, rows in results.items():
        metrics = calculate_metrics(method, rows, len(chunks))
        print(
            f"{method} | {metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | "
            f"{metrics['MRR@10']:.4f} | {metrics['mean_query_ms']:.2f}", flush=True
        )


if __name__ == "__main__":
    main()
