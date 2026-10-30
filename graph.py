from typing import TypedDict, Annotated

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, AIMessage
from qdrant_client import models

from config import llm, COLLECTION_NAME, EMBEDDING_MODEL
from database import client


class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    sources: list
    messages: Annotated[list[AnyMessage], add_messages]


def retrieve_documents(state: ChatState):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=state["question"],
            model=EMBEDDING_MODEL,
        ),
        limit=5, # topk=5,
        with_payload=True,
    )

    context_parts = []
    sources = []

    for rank, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        text = payload.get("page_content", "")
        title = metadata.get("document_title", "Unknown document")
        page = metadata.get("page")

        print(f"\n--- CHUNK {rank} ---")
        print(f"Source: {title}")
        print(f"Page: {page}")
        print(f"Score: {point.score}")
        print("Text:")
        print(text if text else "[No page_content found in payload]")
        print("-" * 70)

        if text:
            context_parts.append(text)

        sources.append({
            "title": title,
            "page": page,
            "score": point.score,
        })

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }



def generate_answer(state: ChatState):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer using the provided document context.
            If the answer is not in the context, say you don't know.

            Context:
            {context}""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | llm).invoke({
        "context": state["context"],
        "messages": state["messages"],
    })

    return {
        "answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }


builder = StateGraph(ChatState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)