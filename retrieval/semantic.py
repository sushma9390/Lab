
from qdrant_client import models
from config import COLLECTION_NAME
from database import client


def semantic_search(query: str, k: int = 4):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=query,
            model="sentence-transformers/all-minilm-l6-v2",
        ),
        limit=k,
        with_payload=True,
    )

    for i, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        print(f"\n--- Result {i} ---")
        print("Content:", payload.get("page_content", ""))
        print("Document:", metadata.get("document_title"))
        print("Page:", metadata.get("page_number"))
        print("Score:", point.score)

    return results.points
