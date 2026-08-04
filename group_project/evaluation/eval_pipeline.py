"""
RAG Evaluation Pipeline — RAGAS.

Đánh giá pipeline RAG (Task 1-10) trên golden_dataset.json với 4 metric RAGAS
chuẩn (faithfulness, answer_relevancy, context_recall, context_precision), so
sánh A/B giữa 2 config retrieval, và xuất báo cáo ra results.md.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Config A "hybrid_rerank": Task 9 đầy đủ — semantic + BM25 → RRF merge → rerank
    → fallback PageIndex nếu cosine gốc < ngưỡng.
Config B "dense_only": chỉ semantic_search (Task 5), bỏ qua BM25/rerank/fallback,
    dùng chung bước reorder + format_context + gọi LLM của Task 10 để so sánh
    công bằng (chỉ khác nhau ở bước retrieval).

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày. Model judge dùng ở đây (gpt-4o-mini qua OpenRouter, KHÔNG phải bản
":free") không tính vào hạn mức 50 req/ngày đó, nhưng vẫn tốn phí theo token.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

CONFIGS = ["hybrid_rerank", "dense_only"]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Chạy RAG pipeline theo từng config
# =============================================================================

def _openrouter_has_credit(api_key: str) -> bool:
    """
    Kiểm tra nhanh (không tốn token) xem key OpenRouter còn credit không, qua
    endpoint /credits — tránh lãng phí gọi chat completion thật rồi mới biết
    bị 402, và tránh phải đoán thứ tự ưu tiên provider mù quáng.
    """
    import requests

    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        data = r.json().get("data", {})
        return data.get("total_credits", 0) - data.get("total_usage", 0) > 0.01
    except Exception:
        return True  # Không check được thì cứ để _call_llm_with_fallback tự lo lỗi runtime


def _candidate_llm_clients():
    """
    Danh sách (client, model) theo thứ tự ưu tiên thử. Provider nào
    _openrouter_has_credit xác nhận còn credit thì đứng trước; nếu không check
    được (lỗi mạng) thì cứ thử OpenRouter trước như mặc định của Task 10.
    Trả về TẤT CẢ candidate hợp lệ (không chỉ 1) vì _call_llm_with_fallback
    vẫn cần phương án dự phòng nếu credit tụt xuống 0 ngay giữa lúc chạy.
    """
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    candidates = []
    # OpenAI đứng đầu: đây là key duy nhất nhóm có và đã verify chạy được.
    # Thiếu nhánh này thì hàm raise "Thiếu OPENROUTER_API_KEY hoặc GEMINI_API_KEY"
    # dù .env có key hợp lệ.
    if openai_key and "..." not in openai_key:
        candidates.append((OpenAI(api_key=openai_key), "gpt-4o-mini"))
    openrouter_ok = bool(openrouter_key) and "..." not in openrouter_key and _openrouter_has_credit(openrouter_key)
    if openrouter_ok:
        candidates.append((
            OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"),
            "openai/gpt-4o-mini",
        ))
    if gemini_key and "..." not in gemini_key:
        candidates.append((
            OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            "gemini-flash-latest",
        ))
    if not openrouter_ok and openrouter_key and "..." not in openrouter_key:
        # Hết credit nhưng vẫn để cuối danh sách phòng khi check nhầm.
        candidates.append((
            OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"),
            "openai/gpt-4o-mini",
        ))
    if not candidates:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY hoặc GEMINI_API_KEY hợp lệ trong .env")
    return candidates


def _call_llm_with_fallback(messages: list[dict], temperature: float, top_p: float) -> str:
    """Gọi LLM, tự rớt xuống candidate tiếp theo nếu provider hiện tại lỗi (hết credit, quota...)."""
    last_error = None
    for client, model in _candidate_llm_clients():
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, top_p=top_p,
            )
            return response.choices[0].message.content
        except Exception as e:  # hết credit (402), quota (429), model không tồn tại (404)...
            last_error = e
            continue
    raise RuntimeError(f"Tất cả LLM candidate đều lỗi. Lỗi cuối: {last_error}")


def run_pipeline(question: str, config: str, top_k: int = 5) -> dict:
    """
    Sinh câu trả lời có citation cho 1 câu hỏi, theo config retrieval chỉ định.

    Dùng chung bước reorder/format_context/prompt của Task 10 cho cả 2 config để
    phần so sánh A/B chỉ phản ánh khác biệt ở retrieval, không lẫn khác biệt
    prompt/model. Việc chọn LLM client tách riêng khỏi Task 10
    (xem _get_generation_client_and_model) để có thể rớt xuống Gemini khi
    OpenRouter hết credit mà không cần sửa code Task 10 của Tuấn.
    """
    from src.task10_generation import (
        format_context,
        reorder_for_llm,
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
    )

    if config == "hybrid_rerank":
        from src.task9_retrieval_pipeline import retrieve

        chunks = retrieve(question, top_k=top_k)
    elif config == "dense_only":
        from src.task5_semantic_search import semantic_search

        chunks = semantic_search(question, top_k=top_k)
        for c in chunks:
            c.setdefault("source", "dense_only")
    else:
        raise ValueError(f"Unknown config: {config}")

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    answer = _call_llm_with_fallback(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return {"answer": answer, "sources": chunks}


# =============================================================================
# RAGAS evaluation
# =============================================================================

def _build_ragas_judge():
    """
    RAGAS mặc định gọi OpenAI trực tiếp (đọc OPENAI_API_KEY) — repo này dùng
    OpenRouter/Gemini làm LLM chính nên phải trỏ judge LLM tường minh, cùng
    logic fallback OpenRouter → Gemini như _get_generation_client_and_model()
    (OpenRouter hết credit giữa buổi thì rớt xuống Gemini, không cần OPENAI_API_KEY).
    Embeddings dùng model local (all-MiniLM-L6-v2, có sẵn qua sentence-transformers)
    để không phụ thuộc thêm provider nào cho phần embedding.
    """
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_ok = bool(openrouter_key) and "..." not in openrouter_key and _openrouter_has_credit(openrouter_key)
    gemini_ok = bool(gemini_key) and "..." not in gemini_key

    # OpenAI trước tiên — key duy nhất nhóm đang có.
    if openai_key and "..." not in openai_key:
        judge_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    elif openrouter_ok:
        judge_llm = ChatOpenAI(
            model="openai/gpt-4o-mini", api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
    elif gemini_ok:
        judge_llm = ChatOpenAI(
            model="gemini-flash-latest", api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif openrouter_key and "..." not in openrouter_key:
        judge_llm = ChatOpenAI(
            model="openai/gpt-4o-mini", api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY hoặc GEMINI_API_KEY hợp lệ trong .env")

    # PHẢI dùng embedding multilingual. RAGAS tính answer_relevancy bằng cách
    # sinh câu hỏi ngược từ answer rồi so embedding với câu hỏi gốc. Golden
    # dataset và answer đều là TIẾNG VIỆT, mà all-MiniLM-L6-v2 chỉ mạnh tiếng
    # Anh -> điểm gần như nhiễu (đo thật: 0.309 / 0.281, trong khi ngưỡng bình
    # thường là 0.7-0.9). bge-m3 đã được cache sẵn từ Task 4, không tải thêm.
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def evaluate_with_ragas(golden_dataset: list[dict], config: str):
    """Chạy pipeline trên golden_dataset với 1 config, evaluate bằng RAGAS."""
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = run_pipeline(item["question"], config=config)
        contexts = [c["content"] for c in result["sources"]] or ["(không có context)"]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    judge_llm, judge_embeddings = _build_ragas_judge()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    return result.to_pandas()


def compare_configs(golden_dataset: list[dict]) -> dict:
    """So sánh A/B giữa 2 config retrieval, trả về {config_name: DataFrame}."""
    return {config: evaluate_with_ragas(golden_dataset, config) for config in CONFIGS}


# =============================================================================
# Export Results
# =============================================================================

def _fmt(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def export_results(comparison: dict, golden_dataset: list[dict]):
    """Format và ghi results.md từ output so sánh A/B."""
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    means = {cfg: {m: df[m].mean() for m in metrics} for cfg, df in comparison.items()}
    for cfg in means:
        means[cfg]["average"] = sum(means[cfg][m] for m in metrics) / len(metrics)

    a, b = CONFIGS[0], CONFIGS[1]

    lines = ["# RAG Evaluation Results", ""]
    # Ghi đúng provider ĐANG dùng thay vì hằng số cứng — trước đây luôn in
    # "gpt-4o-mini qua OpenRouter" kể cả khi thực tế chạy bằng OpenAI.
    _judge_provider = (
        "OpenAI" if os.getenv("OPENAI_API_KEY", "").strip() and "..." not in os.getenv("OPENAI_API_KEY", "")
        else "OpenRouter/Gemini"
    )
    lines += ["## Framework sử dụng", "",
              f"RAGAS 0.1.21 (judge LLM: gpt-4o-mini qua {_judge_provider}, "
              "embeddings: BAAI/bge-m3 local — multilingual, bắt buộc vì golden "
              "dataset và câu trả lời đều là tiếng Việt).", ""]

    lines += ["## Overall Scores", ""]
    lines += [f"| Metric | Config A ({a}) | Config B ({b}) | Δ |",
              "|--------|---------------------------|----------------------|---|"]
    for m in metrics + ["average"]:
        va, vb = means[a][m], means[b][m]
        label = "**Average**" if m == "average" else m
        if va == va and vb == vb:  # not NaN
            diff = f"{va - vb:+.3f}"
        else:
            diff = "N/A"
        lines.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {diff} |")
    lines.append("")

    lines += ["---", "", "## A/B Comparison Analysis", "",
              f"**Config A ({a}):** semantic search + BM25 lexical search → RRF merge → "
              f"rerank → fallback PageIndex nếu cosine gốc < 0.52.", "",
              f"**Config B ({b}):** chỉ semantic search (dense retrieval), bỏ qua BM25/"
              f"rerank/fallback. Cùng bước reorder + format_context + LLM generation "
              f"với Config A để so sánh công bằng, chỉ khác biệt ở retrieval.", ""]

    better = a if means[a]["average"] >= means[b]["average"] else b
    lines += [f"**Kết luận:** `{better}` có average score cao hơn "
              f"({_fmt(means[better]['average'])} so với "
              f"{_fmt(means[a if better == b else b]['average'])}). "
              f"Xem chi tiết per-question để phân tích nguyên nhân trước khi kết luận "
              f"chắc chắn cho cả 2 metric-nhóm (chunk-level vs answer-level).", ""]

    lines += ["---", "", "## Worst Performers (Bottom 3, Config A)", ""]
    df_a = comparison[a].copy()
    df_a["_avg"] = df_a[metrics].mean(axis=1)
    worst = df_a.sort_values("_avg").head(3)
    lines += ["| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |",
              "|---|----------|-------------|-----------|--------|-----------|------------|"]
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        q = str(row["question"])[:60]
        cause = "faithfulness thấp → answer không bám context" if row["faithfulness"] < 0.7 else (
            "context_recall thấp → retriever thiếu evidence" if row["context_recall"] < 0.7 else
            "context_precision thấp → context lẫn nhiều đoạn không liên quan"
        )
        lines.append(
            f"| {i} | {q}... | {_fmt(row['faithfulness'])} | {_fmt(row['answer_relevancy'])} | "
            f"{_fmt(row['context_recall'])} | {_fmt(row['context_precision'])} | {cause} |"
        )
    lines.append("")

    lines += ["---", "", "## Recommendations", "",
              "### Cải tiến 1", "**Action:** Tăng top_k cho các câu có context_recall thấp "
              "(retriever chưa lấy đủ evidence), thử top_k=8 thay vì 5.",
              "**Expected impact:** context_recall tăng, đổi lại context dài hơn, rủi ro "
              "lost-in-the-middle nếu không có reordering tốt.", "",
              "### Cải tiến 2", "**Action:** Với câu có faithfulness thấp, kiểm tra system "
              "prompt có đang bị model \"diễn giải thêm\" ngoài context không — xem lại "
              "ví dụ few-shot trong SYSTEM_PROMPT.",
              "**Expected impact:** answer bám sát context hơn, giảm rủi ro bịa nội dung.", "",
              "### Cải tiến 3", "**Action:** Với context_precision thấp ở dense_only, cân "
              "nhắc thêm bước rerank ngay cả khi không dùng BM25, để lọc bớt chunk gần "
              "đúng ngữ nghĩa nhưng không thực sự liên quan.",
              "**Expected impact:** giảm nhiễu trong context, giúp answer_relevancy ổn định hơn.", ""]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Đã ghi kết quả vào {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset)
    export_results(comparison, golden_dataset)
