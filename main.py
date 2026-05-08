import os
import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

from rag_pipeline import GTURAGPipeline

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip('"' + "'")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip('"' + "'")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
if NVIDIA_API_KEY:
    NVIDIA_API_KEY = NVIDIA_API_KEY.strip('"' + "'")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_PROMPT_LENGTH = 40000

def call_llm(prompt: str, mode: str, max_tokens: int = 2048) -> str:
    """Try Groq first, fallback to Nvidia on rate limit."""
    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {MAX_PROMPT_LENGTH}")
        prompt = prompt[:MAX_PROMPT_LENGTH] + "\n\n[Truncated due to length]"
    
    is_mcq = mode == "mcq"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    if is_mcq:
        payload["response_format"] = {"type": "json_object"}

    if GROQ_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            groq_payload = {**payload, "model": GROQ_MODEL}
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=groq_payload,
                timeout=60
            )
            if response.status_code == 429:
                logger.warning("Groq rate limit hit, trying Nvidia...")
                raise requests.exceptions.HTTPError("429", response=response)
            if response.status_code == 413:
                raise HTTPException(status_code=400, detail=f"Query context too large ({len(prompt)} chars).")
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.warning(f"Groq error: {e}. Trying Nvidia...")
            pass

    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=503, detail="Both Groq and NVIDIA_API_KEY are missing")

    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        nvidia_payload = {
            "model": NVIDIA_MODEL,
            "messages": payload["messages"],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": max_tokens,
        }
        
        if is_mcq:
            nvidia_payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=nvidia_payload,
            timeout=180
        )
        
        if response.status_code == 429:
            raise HTTPException(status_code=503, detail="Nvidia rate limit exceeded")
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.exception(f"Nvidia error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")

# Request Models
class ChatRequest(BaseModel):
    query: str
    mode: str = "tutor"
    marks: int = 0
    subject: str = ""
    web_results: str = ""
    days: int = 0
    hours_per_day: int = 0
    weak_areas: str = ""
    total_questions: int = 0
    difficulty: str = ""
    detail_level: str = ""

class ChatResponse(BaseModel):
    response: str
    detected_intent: str = None
    detected_mode: str = None

# App and Pipeline Initialization (Blocking Startup)
logger.info("Initializing GTU RAG Pipeline...")
pipeline = GTURAGPipeline()
logger.info("Pipeline loaded successfully.")

app = FastAPI(title="GTU RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        logger.info(f"Retrieving context for query: {req.query[:50]}...")
        # Retrieve context from chunks
        contexts = pipeline.retrieve_intelligent(
            req.query, req.subject, 3, req.mode
        )
        logger.info(f"Retrieved {len(contexts)} documents")
        
        # Build prompt using appropriate mode
        prompt = pipeline.generate_prompt(
            req.query, contexts, req.mode,
            **{k: v for k, v in req.model_dump().items() if k not in ['query', 'mode']}
        )
        
        if not GROQ_API_KEY and not OPENROUTER_API_KEY:
            return ChatResponse(response=prompt + "\n\n*(Error: Both GROQ_API_KEY and OPENROUTER_API_KEY are missing from .env)*")

        max_tokens = 4096 if req.mode in ("notes", "coach") else 2048
        result = call_llm(prompt, req.mode, max_tokens)

        return ChatResponse(
            response=result,
            detected_mode=req.mode
        )
        
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return ChatResponse(
            response=f"**Error:**\n```\n{str(e)}\n```"
        )

# Static Files
os.makedirs("static", exist_ok=True)
os.makedirs("static/generated", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Image Download Endpoint
@app.post("/api/notes/download")
async def download_notes(req: ChatRequest):
    try:
        contexts = pipeline.retrieve_intelligent(
            req.query, req.subject, 3, "notes"
        )
        prompt = pipeline.generate_prompt(
            req.query, contexts, "notes",
            **{k: v for k, v in req.model_dump().items() if k not in ['query', 'mode']}
        )
        
        if not GROQ_API_KEY and not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="Both GROQ_API_KEY and OPENROUTER_API_KEY are missing")

        notes_content = call_llm(prompt, "notes", 4096)

        return JSONResponse(content={
            "content": notes_content,
            "download_ready": True
        })
    except Exception as e:
        logger.exception(f"Download notes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
