"""离线检查项目实际模型客户端的设备、输出和显存；不访问知识库或写入索引。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stress", action="store_true", help="同时驻留两个模型，验证长文本批量推理")
    args = parser.parse_args()
    os.environ.update(BGE_DEVICE=args.device, BGE_RERANKER_DEVICE=args.device,
                      BGE_FP16=str(args.device.startswith("cuda")),
                      BGE_RERANKER_FP16=str(args.device.startswith("cuda")),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np
    import torch
    from knowledge.utils.clients.ai_clients import AIClients

    gpu = args.device.startswith("cuda")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError(f"当前解释器 {sys.executable} 的 PyTorch {torch.__version__} 无法使用 CUDA")

    def sync():
        if gpu:
            torch.cuda.synchronize(args.device)

    def measured(fn):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        return result, round((time.perf_counter() - start) * 1000, 2)

    texts = ["高频地波雷达通过平台运动补偿改善测向精度。",
             "向量检索使用语义向量查找相关的文档切片。",
             "极化MUSIC算法通过降维减少谱峰搜索的计算量。",
             "多子问题分别召回证据，再对候选文档重排。"]
    print("Loading embedding model...", flush=True)
    embedder = AIClients.get_bge_m3_client()
    embedder.encode(texts[:1], return_dense=True, return_sparse=True)
    embeddings, embedding_ms = measured(lambda: embedder.encode(texts, return_dense=True, return_sparse=True))
    dense = embeddings["dense_vecs"]
    assert dense.shape == (4, 1024) and np.isfinite(dense).all()
    assert len(embeddings["lexical_weights"]) == 4
    assert all(weights for weights in embeddings["lexical_weights"])
    assert all(np.isfinite(float(v)) for w in embeddings["lexical_weights"] for v in w.values())
    assert str(next(embedder.model.parameters()).device) == args.device

    print("Loading reranker model...", flush=True)
    reranker = AIClients.get_bge_reranker_client()
    pairs = [("如何降低测向算法的搜索复杂度？", text) for text in texts]
    reranker.compute_score(pairs[:1])
    scores, rerank_ms = measured(lambda: reranker.compute_score(pairs))
    assert len(scores) == 4 and np.isfinite(scores).all()
    assert str(next(reranker.model.parameters()).device) == args.device

    result = {
        "python": sys.executable, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if gpu else None,
        "embedding": {"device": str(next(embedder.model.parameters()).device),
                      "dtype": str(next(embedder.model.parameters()).dtype),
                      "batch_size": embedder.batch_size, "shape": list(dense.shape),
                      "dense_and_sparse_valid": True, "warm_4_texts_ms": embedding_ms},
        "reranker": {"device": str(next(reranker.model.parameters()).device),
                     "dtype": str(next(reranker.model.parameters()).dtype),
                     "batch_size": reranker.batch_size, "scores_valid": True, "warm_4_pairs_ms": rerank_ms},
    }
    if args.stress:
        long_texts = [(text + "这是长文本推理测试，用于检查模型的显存占用。") * 100 for text in texts] * 2
        long_embeddings, duration = measured(lambda: embedder.encode(long_texts, return_dense=True, return_sparse=True))
        assert long_embeddings["dense_vecs"].shape == (8, 1024)
        assert np.isfinite(long_embeddings["dense_vecs"]).all()
        result["embedding"]["long_8_texts_ms"] = duration
        long_scores, duration = measured(lambda: reranker.compute_score([(pairs[0][0], text) for text in long_texts]))
        assert len(long_scores) == 8 and np.isfinite(long_scores).all()
        result["reranker"]["long_8_pairs_ms"] = duration
    if gpu:
        result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(args.device) / 1024**2, 1)
        result["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(args.device) / 1024**2, 1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
