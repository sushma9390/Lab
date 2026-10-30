from qdrant_client import models
from database import client
from config import COLLECTION_NAME, EMBEDDING_MODEL


def semantic_search(
    query: str,
    k: int = 4,
    document_title: str | None = None,
):
    query_filter = None

    if document_title:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_title",
                    match=models.MatchValue(
                        value=document_title
                    ),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=query,
            model=EMBEDDING_MODEL,
        ),
        query_filter=query_filter,
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