import os
import sys
import logging
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Set your API keys
try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
except KeyError as e:
    logging.error(f"Missing environment variable: {e}")
    print(f"❌ Missing environment variable: {e}")
    sys.exit(1)

# Connect to your index
index_name = os.environ.get("PINECONE_INDEX", "ragtest")
index = pc.Index(index_name)

# Load chunks from JSON for text retrieval
json_path = os.environ.get("JSON_FILE", r"C:\Users\Swamini\Downloads\azure_ocr_chunks.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logging.info(f"Loaded {len(chunks)} chunks from {json_path}")
except FileNotFoundError:
    logging.error(f"Chunks file not found: {json_path}")
    print(f"❌ Chunks file not found: {json_path}")
    sys.exit(1)

# Create a mapping of chunk_index to text
chunk_text_map = {chunk["chunk_index"]: chunk["text"] for chunk in chunks}

# Build query embedding (must match index dimension = 1536)
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
else:
    query = input("Enter your query: ").strip()

if not query:
    print("❌ No query provided.")
    sys.exit(1)

logging.info(f"Processing query: {query}")
print(f"Query: {query}\n")

try:
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"  # produces 1536-dim embeddings
    )
    query_embedding = response.data[0].embedding
except Exception as e:
    logging.error(f"Error generating query embedding: {e}")
    print(f"❌ Error generating query embedding: {e}")
    sys.exit(1)

# Query Pinecone
try:
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
except Exception as e:
    logging.error(f"Error querying Pinecone: {e}")
    print(f"❌ Error querying Pinecone: {e}")
    sys.exit(1)

# Extract relevant chunk texts with scores
relevant_chunks = []
chunk_scores = []
chunk_indices = []
print(f"\n{'='*80}")
print(f"SEARCH RANKING & RELEVANCE SCORES")
print(f"{'='*80}\n")

for rank, match in enumerate(results.matches, 1):
    chunk_index = match["metadata"].get("chunk_index")
    chunk_text = chunk_text_map.get(chunk_index, "")
    score = match["score"]
    
    if chunk_text:
        relevant_chunks.append(chunk_text)
        chunk_scores.append(score)
        chunk_indices.append(chunk_index)
        
        # Display ranking
        relevance_bar = "█" * int(score * 50) + "░" * (50 - int(score * 50))
        print(f"Rank #{rank} | Chunk {chunk_index} | Score: {score:.4f}")
        print(f"Relevance: [{relevance_bar}]")
        print(f"Preview: {chunk_text[:100].strip()}...")
        print()

# Create context from retrieved chunks
context = "\n\n".join(relevant_chunks)

print(f"{'='*80}")
print(f"GENERATED ANSWER (Based on {len(relevant_chunks)} retrieved chunks)")
print(f"{'='*80}\n")

# Generate answer using OpenAI with retrieved context
system_prompt = """You are a helpful assistant that answers questions based on provided context. 
Answer the user's question using ONLY the information from the context provided.
If the context doesn't contain enough information to answer, say so clearly.
Keep your answer concise and factual."""

user_message = f"""Context:
{context}

Question: {query}

Answer:"""

try:
    response = client.chat.completions.create(
        model="gpt-4",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    answer = response.choices[0].message.content
except Exception as e:
    logging.error(f"Error generating answer: {e}")
    print(f"❌ Error generating answer: {e}")
    sys.exit(1)
print(f"{answer}")

print(f"\n{'='*80}")
print(f"SOURCES & RETRIEVAL SUMMARY")
print(f"{'='*80}")
for i, (chunk_idx, score) in enumerate(zip(chunk_indices, chunk_scores), 1):
    print(f"  [{i}] Chunk {chunk_idx} | Relevance Score: {score:.4f} (Used in answer generation)")

