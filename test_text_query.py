from qdrant_client import models
from database import client
from config import COLLECTION_NAME

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=models.TextQuery(text="soil erosion"),
    using="page_content",
    limit=4,
    with_payload=True,
)

for point in results.points:
    print(point.payload)