# 中文法律知识库 RAG

使用中文法条构建检索知识库，比较 BM25、向量检索和 RRF 混合检索，为后续生成带法条引用的回答提供证据。

**当前已完成数据解析、Milvus Lite 入库和三种检索方式的评测。大模型回答生成、引用核验及问答演示尚未实现。**

- [项目进度与完整实施流程](项目进度.md)
- [检索评测结果 Word 报告](reports/检索评测结果.docx)

## 项目目录

```text
RAG/
├── README.md                    项目入口、目录和运行方法
├── 项目进度.md                  目标、数据、已完成工作和后续计划
├── requirements.txt             Python 依赖
├── parse_corpus.py              法条解析与编号分项分块
├── bm25_baseline.py             中文 BM25 检索与评测
├── vector_baseline.py           BGE 编码、Milvus Lite 入库与向量检索评测
├── hybrid_baseline.py           RRF 融合及三种方法的同轮评测
├── reports/
│   └── 检索评测结果.docx         已完成评测的简明报告
├── STARD/
│   ├── LICENSE                  上游数据许可证
│   └── data/corpus.jsonl        55,348 条原始法条
└── LeCoQA/
    ├── License                  上游数据许可证
    └── data/example/test.json   309 条测试问题和证据标注
```

运行后生成的 `data/processed/` 包含 `chunks.jsonl`、本地向量数据库目录 `legal_vectors.db/` 和导入进度文件 `vector_import_progress.txt`。这些可再生成文件，以及 `.venv`、IDE 配置和缓存，均不提交 Git。

## 使用的数据

数据来自 [STARD](https://github.com/oneal2000/STARD) 与 [LeCoQA](https://github.com/oneal2000/LeCoQA)。两者共用 55,348 条法条和同一组 1,543 个问题；LeCoQA 增加了参考答案。它们不是两组互相独立的问题。

本仓库仅提交运行必需的数据子集：STARD 的一份法条库，以及 LeCoQA 的 309 条测试样本，并保留上游许可证；未提交重复法条库、训练集、负样本和 TSV 示例。原始 `test.json` 是运行输入，包含问题、参考答案及 `match_id` 证据法条 ID，不是可删除的评测结果文件。检索只使用问题文本，`match_id` 仅用于计算指标。

默认每条法条一块；正文有至少两个段首编号分项时，按分项拆块，前置说明并入第一块。每块保留原 `article_id`。当前生成 82,101 个块，不按字符数硬切。

## 运行方法

已验证环境为 Windows、Python 3.12、CPU 推理；Milvus Lite 使用本地数据库，无需额外启动 Docker 服务。首次向量编码需要下载 `BAAI/bge-small-zh-v1.5` 模型，并生成全部法条块的向量，耗时明显长于查询评测。

在项目根目录使用 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe parse_corpus.py
.\.venv\Scripts\python.exe vector_baseline.py
.\.venv\Scripts\python.exe hybrid_baseline.py
```

`vector_baseline.py` 首次生成向量并入库，之后按进度跳过已完成的导入；`hybrid_baseline.py` 复用现有向量库，在同一份测试集上输出 BM25、向量和混合检索的指标表。单独评测 BM25 可运行：

```powershell
.\.venv\Scripts\python.exe bm25_baseline.py
```

评测指标直接打印到终端，不生成逐题结果 JSON 或指标 JSON。已记录的实验结果集中在 Word 报告中。修改数据、分块规则或向量模型后，应重建本地向量库及对应导入进度，不能沿用旧索引。

## 评测口径

同一份 LeCoQA 测试集共 309 题；同一法条有多个块时取最高块分数。混合检索两路各取前 100 个不重复法条，按 `1 / (60 + 排名)` 累加 RRF 分数，返回前 10 条。

Recall@5/10 是每题命中的证据法条数除以该题标注证据总数后取平均；MRR@10 是前 10 名中首个正确法条排名倒数的平均值。耗时不含建索引和模型加载，向量检索包含问题编码，混合检索包含顺序执行两路检索和融合。耗时会随机器和运行状态变化。

当前模型输入上限为 512 token，有 57 个块超过该长度，编码时尾部会被截断，原文仍保留。数据来自既有研究数据集，尚未维护法规时效和版本；当前结果反映检索标注命中情况。
