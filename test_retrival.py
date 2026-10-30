# from retrieval.keyword import keyword_search

# semantic_search(
#     query="What are the environmental effects of cutting down trees?",
#     k=4,
# )

# semantic_search(
#     query="What are the environmental effects of cutting down trees?",
#     k=4,
#     document_title="noise.pdf",
# )

# keyword_search(
#     query="soil erosion",
#     k=4,
# )

from retrieval.hybrid import hybrid_search

hybrid_search(
    query="What are the environmental effects of cutting down trees?",
    k=4,
)