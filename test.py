from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent

# ============================================================
# STEP 1: Load Environment Variables
# ============================================================
# Reads variables from .env file
# Example:
# GROQ_API_KEY=xxxx
# HF_TOKEN=xxxx (optional)
# ============================================================

load_dotenv()

# ============================================================
# STEP 2: Initialize LLM
# ============================================================
# This model is responsible for generating the final answer.
# It DOES NOT read the PDF directly.
# It only receives retrieved chunks from the vector store.
# ============================================================

model = ChatGroq(
    model="qwen/qwen3-32b",
    reasoning_format="parsed"
)

# ============================================================
# STEP 3: Load PDF
# ============================================================
# Reads PDF pages and converts them into LangChain Documents.
# ============================================================

print("Loading PDF Document...")

loader = PyPDFLoader("ai.pdf")
docs = loader.load()

print(f"Pages Loaded: {len(docs)}")

# ============================================================
# STEP 4: Chunking
# ============================================================
# Large documents cannot be embedded efficiently.
# Therefore we split them into smaller pieces (chunks).
#
# chunk_size=1000
# Maximum characters per chunk
#
# chunk_overlap=200
# Repeats 200 characters between chunks
# so context is not lost.
# ============================================================

print("\nCreating Chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_splits = text_splitter.split_documents(docs)

print(f"Total Chunks Created: {len(all_splits)}")

# Show first few chunks to understand chunking
print("\n========== SAMPLE CHUNKS ==========")

for i, chunk in enumerate(all_splits[:3]):
    print(f"\nChunk {i+1}")
    print("-" * 50)
    print(chunk.page_content[:400])

# ============================================================
# STEP 5: Embeddings
# ============================================================
# Converts text into vectors (numbers).
#
# Text:
# "Artificial Intelligence"
#
# becomes:
# [0.124, -0.582, 0.913, ...]
#
# Similar meanings produce similar vectors.
# ============================================================

print("\nBuilding Embeddings...")

# If you have HF_TOKEN in .env, warning disappears
# HF_TOKEN=your_token_here

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# STEP 6: Create Vector Store
# ============================================================
# Stores all chunk embeddings.
#
# Later we search here using similarity search.
# ============================================================

print("Creating Vector Store...")

vector_store = InMemoryVectorStore.from_documents(
    all_splits,
    embeddings
)

# ============================================================
# STEP 7: Retrieval Tool
# ============================================================
# Agent calls this tool when it needs PDF context.
#
# User Question
#      ↓
# Convert question to vector
#      ↓
# Compare against stored vectors
#      ↓
# Return top matching chunks
# ============================================================

@tool
def retrieve_context(query: str):
    """
    Retrieve relevant context from the PDF document.
    """

    print("\n")
    print("=" * 60)
    print("SIMILARITY SEARCH")
    print("=" * 60)

    # Returns document + similarity score
    similar_docs = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    context = []

    for i, (doc, score) in enumerate(similar_docs):

        print(f"\nRetrieved Chunk #{i+1}")
        print(f"Similarity Score: {score}")

        print("\nChunk Preview:")
        print(doc.page_content[:300])

        print("-" * 60)

        context.append(
            f"""
            Content:
            {doc.page_content}

            Source:
            {doc.metadata.get("source", "unknown")}

            Similarity Score:
            {score}
            """
        )

    return "\n\n".join(context)

# ============================================================
# STEP 8: Create Agent
# ============================================================
# Agent decides:
#
# User asks question
#      ↓
# Call retrieve_context()
#      ↓
# Read retrieved chunks
#      ↓
# Generate final answer
# ============================================================

prompt = """
You are a PDF Question Answering Agent.

Always use the retrieve_context tool before answering.

Answer only from the retrieved context.
"""

agent = create_agent(
    model,
    [retrieve_context],
    system_prompt=prompt
)

# ============================================================
# STEP 9: Continuous Question Loop
# ============================================================
# Allows asking multiple questions
# without rebuilding embeddings every time.
#
# Much faster.
# ============================================================

while True:

    print("\n")
    query = input("Ask Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    print("\nGenerating Answer...\n")

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(response["messages"][-1].content)