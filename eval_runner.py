import argparse
import json
import sys
import time
from typing import List, Set
from urllib import request


def normalize_item(text: str) -> str:
    return text.strip()


def to_set(pred) -> Set[str]:
    if pred is None:
        return set()
    if isinstance(pred, list):
        return {normalize_item(str(x)) for x in pred if str(x).strip()}
    if isinstance(pred, str):
        # 常见分隔符拆分列表
        parts: List[str] = []
        for sep in ["、", ",", "，", ";", "；", "|"]:
            if sep in pred:
                parts = [p for p in pred.split(sep)]
                break
        if not parts:
            parts = [pred]
        return {normalize_item(p) for p in parts if normalize_item(p)}
    return {normalize_item(str(pred))}


def post_json(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    # 放宽超时时间，避免大模型响应过慢导致评测中断
    with request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def evaluate(dataset_path: str, endpoint: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    right = 0
    f1_sum = 0.0
    hallucination_count = 0

    per_sample = []

    for item in dataset:
        q = item["question"].strip()
        gold = to_set(item["answers"])

        start = time.time()
        res = post_json(endpoint, {"question": q})
        elapsed = (time.time() - start) * 1000

        # 兼容 /query_v2（final_answer 为主）和 /query（data 为主）
        if "final_answer" in res:
            pred_set = to_set(res.get("final_answer"))
            # 如果 final_answer 为空但有 kg_answers，用 kg_answers
            if not pred_set and "kg_answers" in res:
                pred_set = to_set(res.get("kg_answers"))
        else:
            pred_set = to_set(res.get("data"))

        inter = pred_set & gold
        prec = len(inter) / len(pred_set) if pred_set else 0.0
        rec = len(inter) / len(gold) if gold else (0.0 if pred_set else 1.0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        correct = pred_set == gold if gold else (len(pred_set) == 0)
        if correct:
            right += 1
        f1_sum += f1

        # 简单幻觉：预测里有不在 gold 的内容（且 gold 非空时才计）
        hallucinated = bool(pred_set - gold) and bool(gold)
        if hallucinated:
            hallucination_count += 1

        per_sample.append(
            {
                "question": q,
                "pred": list(pred_set),
                "gold": list(gold),
                "prec": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "elapsed_ms": round(elapsed, 1),
                "raw_response": res,
            }
        )

    acc = right / total if total else 0.0
    macro_f1 = f1_sum / total if total else 0.0
    hallucination_rate = hallucination_count / total if total else 0.0

    return {
        "total": total,
        "accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "samples": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA outputs against gold answers.")
    parser.add_argument("--data", default="eval_dataset.json", help="Dataset JSON path")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5001/query_v2",
        help="HTTP endpoint for QA API",
    )
    args = parser.parse_args()

    try:
        report = evaluate(args.data, args.endpoint)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("总样本数:", report["total"])
    print("准确率:", report["accuracy"])
    print("宏平均 F1:", report["macro_f1"])
    print("幻觉率:", report["hallucination_rate"])
    print("\n逐条结果:")
    for s in report["samples"]:
        print(
            f"- Q: {s['question']} | pred={s['pred']} | gold={s['gold']} | "
            f"prec={s['prec']} rec={s['recall']} f1={s['f1']} | {s['elapsed_ms']} ms"
        )


if __name__ == "__main__":
    main()
