import os
import tempfile
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from database import client
from config import COLLECTION_NAME, EMBEDDING_MODEL


def ingest_document(file, filename):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata = {
            "document_title": filename,
            "page_number": doc.metadata["page"] + 1,
            "source": filename,
    }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(docs)

        points = []

        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=models.Document(
                        text=chunk.page_content,
                        model=EMBEDDING_MODEL,
                    ),
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        if points:
            client.upload_points(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

        return len(chunks)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)