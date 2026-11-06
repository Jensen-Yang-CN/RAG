"""解析 STARD/data/corpus.jsonl，按法条及编号分项生成检索块。"""

import json
import re
from pathlib import Path


ROOT = Path(__file__).parent
SOURCE = ROOT / "STARD" / "data" / "corpus.jsonl"
OUTPUT = ROOT / "data" / "processed"
ITEM_START = re.compile(r"^(?:[（(][一二三四五六七八九十百\d]+[）)]|[一二三四五六七八九十\d]+[、.．])")


def split_content(content):
    """默认一条法条一块；有多个编号分项时才按分项拆块。"""
    paragraphs = [part.strip() for part in content.split("\n\n") if part.strip()]
    starts = [i for i, part in enumerate(paragraphs) if ITEM_START.match(part)]
    if len(starts) < 2:
        return [content]

    chunks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(paragraphs)
        if i == 0:
            start = 0  # 法条开头的说明并入第一个分项
        chunks.append("\n\n".join(paragraphs[start:end]))
    return chunks


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chunks_path = OUTPUT / "chunks.jsonl"
    article_count = 0
    chunk_count = 0

    with SOURCE.open(encoding="utf-8") as source, chunks_path.open("w", encoding="utf-8") as chunks_file:
        for line in source:
            row = json.loads(line)
            article_id = row["id"]
            name = row["name"].strip()
            content = row["content"].replace("\\n", "\n").strip()

            article_count += 1

            for part, text in enumerate(split_content(content)):
                chunk = {
                    "chunk_id": f"{article_id}_{part}",
                    "article_id": article_id,
                    "name": name,
                    "content": text,
                }
                chunks_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(f"读取法条：{article_count} 条")
    print(f"检索块：{chunk_count} 块 -> {chunks_path}")


if __name__ == "__main__":
    main()
