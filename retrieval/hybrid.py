from retrieval.semantic import semantic_search
from retrieval.keyword import keyword_search


def hybrid_search(query: str, k: int = 4):
    dense_results = semantic_search(query=query, k=k * 2)
    keyword_results = keyword_search(query=query, k=k * 2)

    scores = {}
    points = {}

    # RRF: combine rankings, not the original scores.
    for result_list in (dense_results, keyword_results):
        for rank, point in enumerate(result_list, start=1):
            point_id = str(point.id)
            points[point_id] = point
            scores[point_id] = (
                scores.get(point_id, 0) + 1 / (60 + rank)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:k]

    print("\n========== HYBRID RESULTS ==========")

    for rank, point_id in enumerate(ranked_ids, start=1):
        point = points[point_id]
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        print(f"\n--- Result {rank} ---")
        print("Content:", payload.get("page_content", ""))
        print("Document:", metadata.get("document_title"))
        print("Page:", metadata.get("page_number"))
        print("RRF score:", round(scores[point_id], 5))

    return [points[point_id] for point_id in ranked_ids]