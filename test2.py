from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# ==================================================
# STEP 1 : LOAD LLM
# ==================================================

model = ChatGroq(
    model="qwen/qwen3-32b",
    reasoning_format="parsed"
)

# ==================================================
# STEP 2 : LOAD PDF
# ==================================================

print("\nLoading PDF...")

loader = PyPDFLoader("ai.pdf")
docs = loader.load()

print(f"Pages Loaded: {len(docs)}")

# ==================================================
# STEP 3 : CHUNKING
# ==================================================
#
# Large documents are split into smaller pieces.
#
# chunk_size=1000
# chunk_overlap=200
#
# Example:
#
# Chunk 1 : 1-1000 chars
# Chunk 2 : 800-1800 chars
#
# Overlap prevents context loss.
#
# ==================================================

print("\nCreating Chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)

print(f"Total Chunks: {len(all_splits)}")

print("\n========== SAMPLE CHUNKS ==========")

for i, chunk in enumerate(all_splits[:3]):

    print(f"\nChunk {i+1}")
    print("-" * 50)

    print(chunk.page_content[:400])

# ==================================================
# STEP 4 : EMBEDDINGS
# ==================================================
#
# Text -> Vector
#
# "Artificial Intelligence"
#
# becomes:
#
# [0.22, -0.11, 0.88, ...]
#
# Similar meaning => similar vectors
#
# ==================================================

print("\nBuilding Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# SHOW SAMPLE VECTOR
# ==================================================

sample_text = all_splits[0].page_content

sample_vector = embeddings.embed_query(sample_text)

print("\n========== CHUNK VECTOR EXAMPLE ==========")

print(f"Vector Dimension: {len(sample_vector)}")

print("\nFirst 20 Values:")

print(sample_vector[:20])

# ==================================================
# STEP 5 : VECTOR STORE
# ==================================================
#
# Stores vectors.
#
# Used later for similarity search.
#
# ==================================================

print("\nCreating Vector Store...")

vector_store = InMemoryVectorStore.from_documents(
    all_splits,
    embeddings
)

print("Vector Store Ready")

# ==================================================
# QUESTION LOOP
# ==================================================

while True:

    print("\n")
    print("=" * 70)

    query = input("Ask Question (exit to quit): ")

    if query.lower() == "exit":
        break

    # ==========================================
    # QUERY VECTOR
    # ==========================================

    query_vector = embeddings.embed_query(query)

    print("\n========== QUERY VECTOR ==========")

    print(f"Dimension: {len(query_vector)}")

    print("\nFirst 20 Values:")

    print(query_vector[:20])

    # ==========================================
    # SIMILARITY SEARCH
    # ==========================================
    #
    # Convert query to vector
    #
    # Compare against all chunk vectors
    #
    # Return best matching chunks
    #
    # ==========================================

    print("\nSearching Similar Chunks...")

    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    print("\n========== TOP MATCHING CHUNKS ==========")

    context_parts = []

    for i, (doc, score) in enumerate(results):

        print(f"\nChunk Rank #{i+1}")

        print(f"Similarity Score: {score}")

        # show chunk vector

        chunk_vector = embeddings.embed_query(
            doc.page_content
        )

        print(
            f"Chunk Vector First 10 Values:"
        )

        print(chunk_vector[:10])

        print("\nChunk Preview:")

        print(doc.page_content[:400])

        print("-" * 60)

        context_parts.append(doc.page_content)

    # ==========================================
    # CONTEXT SENT TO LLM
    # ==========================================

    context = "\n\n".join(context_parts)

    print("\n========== CONTEXT SENT TO LLM ==========")

    print(context[:1000])

    # ==========================================
    # FINAL PROMPT
    # ==========================================

    prompt = f"""
Answer ONLY using the provided context.

CONTEXT:
{context}

QUESTION:
{query}
"""

    # ==========================================
    # LLM CALL
    # ==========================================

    print("\nGenerating Answer...")

    response = model.invoke(prompt)

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(response.content)

    print("\n")
    print("=" * 70)
    print("RAG FLOW")
    print("=" * 70)

    print("""
1. PDF Loaded

2. PDF Split Into Chunks

3. Chunks Converted Into Vectors

4. Query Converted Into Vector

5. Query Compared With Chunk Vectors

6. Top Matching Chunks Retrieved

7. Retrieved Context Sent To LLM

8. LLM Generated Final Answer
""")