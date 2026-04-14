import os
import json
import logging
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import pinecone

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_env() -> None:
    project_root = os.path.dirname(__file__)
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path)
    logging.info("Environment variables loaded.")


def init_clients():
    """Initialize OpenAI and Pinecone clients for the new Pinecone SDK (v8+)."""
    index_name = os.environ.get("PINECONE_INDEX", "ragtest")

    pc = pinecone.Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
    )

    # Check if index exists and has correct dimension
    try:
        index_info = pc.describe_index(index_name)
        if index_info.dimension != 1536:
            logging.error(f"Index '{index_name}' has dimension {index_info.dimension}, but embeddings are 1536-dimensional.")
            print(f"ERROR: Index '{index_name}' has dimension {index_info.dimension}, but embeddings are 1536-dimensional.")
            print(f"Please delete the index and recreate it with dimension 1536:")
            print(f"  pc.delete_index('{index_name}')")
            print(f"  pc.create_index(name='{index_name}', dimension=1536, metric='cosine')")
            raise ValueError(f"Index dimension mismatch: {index_info.dimension} != 1536")
        logging.info(f"Index '{index_name}' exists with correct dimension.")
    except Exception as e:
        if "not found" in str(e).lower():
            logging.info(f"Index '{index_name}' does not exist. Creating it...")
            print(f"Index '{index_name}' does not exist. Creating it...")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine"
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            logging.error(f"Error initializing Pinecone: {e}")
            raise

    index = pc.Index(host=pc.describe_index(index_name).host)
    return index


def embed_text(text: str, client: OpenAI) -> List[float]:
    """Generate embedding for text using OpenAI API (v1.0.0+)."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def load_chunks(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vectors(chunks: List[Dict], source_name: str, client: OpenAI) -> List[tuple]:
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{chunk['chunk_index']}"
        text = chunk["text"]
        
        # Generate embedding
        embedding = embed_text(text, client)
        print(f"  Chunk {i+1}/{len(chunks)}: embedding dimension = {len(embedding)}")
        
        metadata = {
            "chunk_index": chunk.get("chunk_index"),
            "token_count": chunk.get("token_count"),
            "char_start": chunk.get("char_start"),
            "char_end": chunk.get("char_end"),
            "clause_count": chunk.get("clause_count"),
            "source": source_name,
        }
        vectors.append((chunk_id, embedding, metadata))
    return vectors


def upsert_in_batches(index, vectors: List[tuple], batch_size: int = 50) -> None:
    """Upsert vectors into Pinecone index in batches."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        vector_list = [
            {
                "id": v[0],
                "values": v[1],
                "metadata": v[2],
            }
            for v in batch
        ]
        try:
            index.upsert(vectors=vector_list)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
        except Exception as e:
            print(f"Error upserting batch {i // batch_size + 1}: {e}")
            raise


def main() -> None:
    load_env()
    logging.info("Initializing clients...")
    try:
        index = init_clients()
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except KeyError as e:
        logging.error(f"Missing environment variable: {e}")
        print(f"❌ Missing environment variable: {e}")
        return

    json_path = os.environ.get("JSON_FILE", r"C:\Users\Swamini\Downloads\azure_ocr_chunks.json")
    source_name = os.environ.get("SOURCE_NAME", "source-document")

    try:
        chunks = load_chunks(json_path)
        logging.info(f"Loaded {len(chunks)} chunks from {json_path}")
        print(f"Loaded {len(chunks)} chunks from {json_path}")
    except FileNotFoundError:
        logging.error(f"Chunks file not found: {json_path}")
        print(f"❌ Chunks file not found: {json_path}")
        return

    logging.info("Building vectors and generating embeddings...")
    print("Building vectors and generating embeddings...")
    try:
        vectors = build_vectors(chunks, source_name, client)
        logging.info(f"Generated {len(vectors)} embeddings")
        print(f"Generated {len(vectors)} embeddings")
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        print(f"❌ Error generating embeddings: {e}")
        return

    logging.info("Upserting vectors to Pinecone...")
    print("Upserting vectors to Pinecone...")
    try:
        upsert_in_batches(index, vectors)
        logging.info(f"Successfully upserted {len(vectors)} vectors into Pinecone index '{os.environ.get('PINECONE_INDEX', 'ragtest')}'.")
        print(f"Done. Upserted {len(vectors)} vectors into Pinecone index '{os.environ.get('PINECONE_INDEX', 'ragtest')}'.")
    except Exception as e:
        logging.error(f"Error upserting vectors: {e}")
        print(f"❌ Error upserting vectors: {e}")


if __name__ == "__main__":
    main()
