import csv
import json
import re
import time
from pathlib import Path

from qdrant_client import models
from config import COLLECTION_NAME, EMBEDDING_MODEL
from database import client


# Change these values for each experiment.
EXPECTED_COLLECTION = "rag_m2_reindex_500c_100o_minilm"

DATASET_PATH = Path("rag_module2_golden_dataset.jsonl")
RESULTS_PATH = Path("rag_m2_reindex_500c_100o_minilm_results.csv")
SUMMARY_PATH = Path("rag_m2_reindex_500c_100o_minilm_summary.json")

TOP_K = 5


def normalize_source(value):
    """Ignore file extensions, spaces, underscores, and punctuation."""
    name = Path(str(value or "")).stem.lower()
    return re.sub(r"[^a-z0-9]", "", name)


def source_matches(expected, actual):
    """Match document names after normalization."""
    expected = normalize_source(expected)
    actual = normalize_source(actual)
    return bool(expected and actual and expected == actual)


def main():
    # Prevent accidentally evaluating the wrong collection.
    if COLLECTION_NAME != EXPECTED_COLLECTION:
        raise ValueError(
            f"COLLECTION_NAME in config.py is {COLLECTION_NAME!r}, "
            f"but this experiment expects {EXPECTED_COLLECTION!r}."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH.resolve())

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    result_rows = []
    question_results = []

    for index, item in enumerate(questions, start=1):
        question_id = str(item["question_id"])
        question = item["question"]
        expected_source = item.get("source_document", "")
        answerable = bool(item.get("answerable", True))

        start = time.perf_counter()

        response = client.query_points(
            collection_name=EXPECTED_COLLECTION,
            query=models.Document(
                text=question,
                model=EMBEDDING_MODEL,
            ),
            limit=TOP_K,
            with_payload=True,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        points = response.points

        retrieved_chunks = []

        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            retrieved_source = metadata.get("document_title", "")
            retrieved_text = payload.get("page_content", "")

            matched = source_matches(
                expected_source, retrieved_source
            )

            retrieved_chunks.append({
                "rank": rank,
                "source": retrieved_source,
                "source_match": matched,
                "text": retrieved_text,
            })

            result_rows.append({
                "question_id": question_id,
                "question": question,
                "answerable": answerable,
                "expected_source": expected_source,
                "rank": rank,
                "retrieved_source": retrieved_source,
                "page": metadata.get("page"),
                "score": point.score,
                "source_match": matched if answerable else "",
                "retrieved_text": retrieved_text,
                "latency_ms": round(latency_ms, 2),
            })

        source_hit = (
            any(chunk["source_match"] for chunk in retrieved_chunks)
            if answerable else None
        )

        question_results.append({
            "question_id": question_id,
            "answerable": answerable,
            "source_hit_at_5": source_hit,
            "retrieved_count": len(points),
            "retrieval_latency_ms": round(latency_ms, 2),
        })

        print(
            f"[{index}/{len(questions)}] {question_id}: "
            f"{len(points)} chunks, {latency_ms:.2f} ms"
        )

    fields = [
        "question_id", "question", "answerable",
        "expected_source", "rank", "retrieved_source",
        "page", "score", "source_match",
        "retrieved_text", "latency_ms",
    ]

    with RESULTS_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    answerable_results = [
        row for row in question_results if row["answerable"]
    ]
    source_hits = [
        row for row in answerable_results
        if row["source_hit_at_5"]
    ]

    summary = {
        "collection": EXPECTED_COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "total_questions": len(questions),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": (
            len(questions) - len(answerable_results)
        ),
        "source_hit_at_5": (
            len(source_hits) / len(answerable_results)
            if answerable_results else None
        ),
        "average_retrieval_latency_ms": (
            round(
                sum(row["retrieval_latency_ms"]
                    for row in question_results)
                / len(question_results),
                2,
            ) if question_results else None
        ),
        "questions": question_results,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n===== RETRIEVAL RUN COMPLETE =====")
    print(f"Collection: {EXPECTED_COLLECTION}")
    print(f"Questions: {len(questions)}")
    print(f"Chunks saved: {len(result_rows)}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

    if answerable_results:
        print(
            f"Source Hit Rate@{TOP_K}: "
            f"{len(source_hits)}/{len(answerable_results)} "
            f"({len(source_hits) / len(answerable_results):.2%})"
        )

    print(
        "Average retrieval latency: "
        f"{summary['average_retrieval_latency_ms']} ms"
    )


if __name__ == "__main__":
    main()