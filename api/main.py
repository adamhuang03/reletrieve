from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from semantic_chunkers import StatisticalChunker
from semantic_router import Route
from semantic_router.encoders import HFEndpointEncoder, HuggingFaceEncoder
import logging
from dotenv import load_dotenv
import os
import re
from typing import List

# Load environment variables
load_dotenv(".env.local")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is missing.")

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():  # Avoid adding handlers multiple times
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RelevancyRequest(BaseModel):
    query: str
    text: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}

def cosine_similarity_numpy(a, b): 
    # Convert inputs to numpy arrays and ensure correct shape
    a = np.array(a[0] if isinstance(a, list) else a)  # HF endpoint returns list of lists
    b = np.array(b[0] if isinstance(b, list) else b)
    
    # Add batch dimension if needed
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)
    
    # Calculate cosine similarity
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return np.dot(a, b.T) / np.outer(norm_a, norm_b)

def preprocess_text(text):
    # Check for areas where \\n repeat, something like '\\n\\n\\n\\n\\n\\n\\n\\n'.
    # If they repeat more than twice, keep only 2 and remove excess
    text = re.sub(r'(\\n){3,}', '\\n\\n', text)
    return text

# uvicorn main:app --log-level info --reload    
@app.post("/relevancy")
async def calculate_relevancy(request: RelevancyRequest):
    # try:
        logger.info("Relevancy request received.")

        # Preprocess the text
        text = preprocess_text(request.text)

        class CustomHFEncoder(HFEndpointEncoder):
            def __call__(self, docs: List[str]):
                # Get embeddings from parent class
                embeddings = super().__call__(docs)
                # Convert to numpy and squeeze extra dimension
                embeddings = np.array(embeddings)
                if len(embeddings.shape) == 3:
                    embeddings = embeddings.squeeze(1)
                return embeddings

        # Initialize the encoder with our custom class
        encoder = CustomHFEncoder(
            huggingface_api_key=API_KEY,
            huggingface_url="https://ukapy1uoys1zmu4x.us-east-1.aws.endpoints.huggingface.cloud"
        )
        # encoder = HuggingFaceEncoder(
        #     name="sentence-transformers/all-MiniLM-L6-v2",
        # )

        # Test the encoder
        logger.info("Testing encoder...")
        test_result = encoder(["Test sentence"])
        logger.info(f"Encoder test result shape: {np.array(test_result).shape}")

        chunker = StatisticalChunker(
            encoder=encoder,
            min_split_tokens=10,
            max_split_tokens=250,  # Further reduced to be safe
            # split_tokens_tolerance=5,
            dynamic_threshold=True,
            window_size=4,
            enable_statistics=True,
            plot_chunks=False
        )
        # logger.info("Processing text: %s", json.loads(request.text)[:200] + "...")
        # logger.info("Processing text: %s", json.loads(request.text))
        
        # Process the cleaned text
        chunks_w_format = chunker(docs=[text])
        # chunker.print(chunks_w_format[0])

        query_embedding = encoder(docs=["I want to learn about " + request.query])
        logger.info(f"Query embedding shape: {np.array(query_embedding).shape}")
        chunk_texts = [chunk.content for chunk in chunks_w_format[0]]
        logger.info(f"Number of chunks: {len(chunk_texts)}")
        logger.info(f"Chunk texts: {chunk_texts[:5]}")
        chunk_embeddings = encoder(docs=chunk_texts)

        # # Log shapes for debugging
        # logger.info(f"Query embedding type: {type(query_embedding)}")
        # logger.info(f"Query embedding shape: {np.array(query_embedding).shape}")
        # logger.info(f"Chunk embeddings type: {type(chunk_embeddings)}")
        # logger.info(f"Chunk embeddings shape: {np.array(chunk_embeddings).shape}")
        # logger.info(f"Number of chunks: {len(chunk_texts)}")

        # # 5. Find most similar chunks using cosine similarity
        similarities = cosine_similarity_numpy(query_embedding, chunk_embeddings)
        top_k = 5
        top_indices = similarities[0].argsort()[-top_k:][::-1]  # Sort in descending order
        top_similarities = similarities[0][top_indices]

        # Print the top 3 chunks with their similarity scores
        BASE_SIMILARITY_VALUE = 0.25
        SIMILARITY_AMPLIFIER = 4
        SIMILARITY_HIGH_FACTOR = 1.5
        SIMILARITY_THRESHOLD_HIGH = 0.8
        SIMILARITY_THRESHOLD = 0.5
        SUM_SIMILARITY = len(similarities[0])
        total_similarity = sum([
            similarity * SIMILARITY_AMPLIFIER * SIMILARITY_HIGH_FACTOR 
            if similarity >= SIMILARITY_THRESHOLD_HIGH
            else 
            similarity * SIMILARITY_AMPLIFIER 
            if similarity >= SIMILARITY_THRESHOLD
            else similarity
            for similarity in similarities[0]
            ]) + BASE_SIMILARITY_VALUE * SUM_SIMILARITY
        avg_similarity = min(0.99, total_similarity / SUM_SIMILARITY)
        logger.info(f'{total_similarity}/{SUM_SIMILARITY}')
        logger.info(f'Average similarity: {avg_similarity}')
        similarity = round(avg_similarity, 2)
        return {
            "relevancy_score": similarity*100,
            "top_chunks": [
                {
                    "text": chunks_w_format[0][i],
                    "similarity": round(similarities[0][i], 2)
                } for i in top_indices
            ],
            "status": "success"
        }
        # return {"success": True}
    # except Exception as e:
    #     return {
    #         "status": "error",
    #         "message": str(e)
    #     }
    #     # return {"success": True}
