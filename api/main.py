from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
from semantic_chunkers import StatisticalChunker
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

@app.post("/relevancy")
async def calculate_relevancy(request: RelevancyRequest):
    try:
        logger = logging.getLogger(__name__)

        # Initialize encoder with HuggingFace API
        encoder = HuggingFaceEncoder(
            api_key="",  # Will use free tier if no API key provided
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        chunker = StatisticalChunker(
            encoder=encoder,
            min_split_tokens=10,
            # max_split_tokens=100,
            dynamic_threshold=True,
            window_size=4,
            enable_statistics=True,
            plot_chunks=False
        )

        logger.info("Test")
        # logger.info("Processing text: %s", json.loads(request.text)[:200] + "...")
        # logger.info("Processing text: %s", json.loads(request.text))
        
        chunks_w_format = chunker(docs=[json.loads(request.text)])
        chunker.print(chunks_w_format[0])

        query_embedding = encoder(["I want to learn about " + request.query])
        chunk_texts = [chunk.splits[0] for chunk in chunks_w_format[0]]
        chunk_embeddings = encoder(chunk_texts)

        # 5. Find most similar chunks using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, chunk_embeddings)
        top_k = 5
        top_indices = similarities[0].argsort()[-top_k:][::-1]  # Sort in descending order
        top_similarities = similarities[0][top_indices]

        # Print the top 3 chunks with their similarity scores
        BASE_SIMILARITY_VALUE = 0.25
        SIMILARITY_AMPLIFIER = 9
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
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
        # return {"success": True}
