
from database import client
from config import COLLECTION_NAME
from qdrant_client import models

# Check the collection before deleting anything
info = client.get_collection(COLLECTION_NAME)
print("Collection:", COLLECTION_NAME)
print("Points before cleanup:", info.points_count)

confirm = input(
    "Delete ALL points in this collection? Type YES to continue: "
)

if confirm == "YES":
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[])
        ),
        wait=True,
    )

    info = client.get_collection(COLLECTION_NAME)
    print("Points after cleanup:", info.points_count)
    print("Collection configuration preserved.")
else:
    print("Cleanup cancelled.")
