from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from ingestion import ingest_document
from graph import builder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from uuid import uuid4
import os

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = InMemorySaver()

    app.state.chat_graph = builder.compile(
        checkpointer=checkpointer
    )

    yield


app = FastAPI(
    title="Local RAG Chatbot",
    lifespan=lifespan,
)

class ChatRequest(BaseModel):
    question: str
    thread_id : str | None = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        chunks = ingest_document(content, file.filename)

        return {
            "message": "Document processed successfully",
            "filename": file.filename,
            "chunks_stored": chunks
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}"
        ) from exc


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    thread_id = body.thread_id or str(uuid4())
    config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
    try:

        result =  request.app.state.chat_graph.invoke({
            "messages": [HumanMessage(content=body.question)],
            "question": body.question,
            "context": "",
            "answer": "",
            },config=config)

        return {"answer": result["answer"],
                "thread_id": thread_id,
                "sources": result.get("sources", []),
                }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}"
        ) from exc