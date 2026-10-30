from qdrant_client import models

from database import client
from config import COLLECTION_NAME


def keyword_search(query: str, k: int = 4):
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="page_content",
                    match=models.MatchText(text=query),
                )
            ]
        ),
        limit=k,
        with_payload=True,
    )

    for i, point in enumerate(results, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        print(f"\n--- Result {i} ---")
        print("Content:", payload.get("page_content", ""))
        print("Document:", metadata.get("document_title"))
        print("Page:", metadata.get("page_number"))

    return results