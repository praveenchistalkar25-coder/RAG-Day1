# Azure OCR RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) system for processing OCR text from images, chunking semantically, embedding, and querying with grounded answers.

## Features
- **OCR Extraction**: Uses Azure Computer Vision to extract text from images, handling multi-column layouts.
- **Semantic Chunking**: Splits text into balanced chunks with overlap, limited to 10 chunks max.
- **Vector Storage**: Embeds chunks using OpenAI and stores in Pinecone.
- **RAG Querying**: Searches for relevant chunks and generates GPT-4 answers with ranking transparency.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Set up `.env` file with your API keys:
   ```
   AZURE_KEY=your-azure-key
   AZURE_ENDPOINT=your-azure-endpoint
   OPENAI_API_KEY=your-openai-key
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_INDEX=ragtest
   IMAGE_PATH=path/to/image.png
   JSON_FILE=path/to/chunks.json
   # Optional: NUM_COLUMNS, CHUNK_SIZE_TOKENS, etc.
   ```

## Usage
1. **Chunk the image**: `python chunk.py`
2. **Ingest to Pinecone**: `python pinecone_ingest.py`
3. **Query**: `python query.py "Your question here"` or run `python query.py` and enter query interactively.

## Configuration
All settings can be customized via environment variables in `.env`.

## Effectiveness
- Handles complex newspaper layouts with column sorting.
- Produces 9-10 balanced chunks for optimal retrieval.
- Provides transparent ranking and grounded answers.
- Robust error handling and logging.
