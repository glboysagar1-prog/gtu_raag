import os
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# Import the RAG Pipeline
from rag_pipeline import GTURAGPipeline, classify_intent_fast, INTENT_TO_MODE

# ============================================
# Configuration
# ============================================
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip('"' + "'")

# Model configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HANDWRITTEN_ANSWER = os.getenv("HANDWRITTEN_ANSWER", "true").lower() == "true"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# Rate Limiter Setup
# ============================================
limiter = Limiter(key_func=get_remote_address)

# ============================================
# Conversation History Store (in-memory)
# ============================================
class ConversationStore:
    """Simple in-memory conversation store for session-based history."""

    def __init__(self, max_history: int = 10):
        self._store: dict[str, list[dict]] = {}
        self._max_history = max_history

    def get(self, session_id: str) -> list[dict]:
        return self._store.get(session_id, [])

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})
        # Trim to max history
        self._store[session_id] = self._store[session_id][-self._max_history:]

    def clear(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]

conversation_store = ConversationStore()

# ============================================
# Request/Response Models
# ============================================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question or query")
    mode: str = Field(default="auto", pattern="^(auto|tutor|coach|mcq|notes|flow)$", description="Response mode")
    marks: int = Field(default=0, ge=0, le=10, description="Marks for tutor mode (3, 4, 7, 10)")
    subject: str = Field(default="", max_length=200, description="Subject name")
    web_results: str = Field(default="", description="Optional web search results")
    days: int = Field(default=0, ge=1, le=30, description="Days for study plan")
    hours_per_day: int = Field(default=0, ge=1, le=12, description="Hours per day for study plan")
    weak_areas: str = Field(default="", max_length=500, description="Weak areas for study plan")
    total_questions: int = Field(default=0, ge=1, le=15, description="Number of MCQs")
    difficulty: str = Field(default="", pattern="^(easy|medium|hard)$", description="Difficulty for MCQs")
    detail_level: str = Field(default="", pattern="^(beginner|intermediate|advanced|brief|standard|detailed)$", description="Detail level")
    user_expertise: str = Field(default="beginner", pattern="^(beginner|intermediate|advanced)$", description="User expertise for flow mode")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")

    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    detected_intent: Optional[str] = None
    detected_mode: Optional[str] = None
    session_id: Optional[str] = None
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    pipeline_loaded: bool

# ============================================
# FastAPI Lifespan Events (Async Initialization)
# ============================================
pipeline: Optional[GTURAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan context manager for proper startup/shutdown."""
    global pipeline

    logger.info("Starting GTU RAG Pipeline...")
    try:
        # Run blocking pipeline initialization in a thread to not block startup
        pipeline = await asyncio.get_event_loop().run_in_executor(
            None, GTURAGPipeline
        )
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        pipeline = None

    yield

    # Cleanup on shutdown
    logger.info("Shutting down GTU RAG Pipeline...")
    if pipeline and hasattr(pipeline, 'chroma_client'):
        pipeline.chroma_client = None

app = FastAPI(
    title="GTU RAG System",
    description="Educational Q&A System with hybrid retrieval and multi-mode responses",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Request Logging Middleware
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for monitoring and debugging."""
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code}")
    return response

# ============================================
# LLM Call with Retry Logic
# ============================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def call_groq_with_retry(messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, json_mode: bool = False) -> dict:
    """Call Groq API with exponential backoff retry logic."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()

# ============================================
# API Endpoints
# ============================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=GROQ_MODEL,
        pipeline_loaded=pipeline is not None
    )

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute per IP
async def chat_endpoint(request: Request, req: ChatRequest):
    """Main chat endpoint for all modes."""
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Generate or use session ID
        session_id = req.session_id or f"session_{id(req)}_{asyncio.get_event_loop().time()}"

        # Store user query in history
        conversation_store.add(session_id, "user", req.query)

        # Auto-classify intent if mode is "auto"
        detected_intent = None
        detected_mode = None

        if req.mode == "auto":
            detected_intent = classify_intent_fast(req.query)
            req.mode = INTENT_TO_MODE.get(detected_intent, "tutor")
            detected_mode = req.mode
            logger.info(f"Auto-detected intent: {detected_intent} → {req.mode}")

        # Retrieve context from chunks using Intelligent Retrieval
        logger.info(f"Retrieving context for query: {req.query[:50]}...")
        contexts = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline.retrieve_intelligent,
            req.query,
            req.subject,
            3,
            req.mode
        )
        logger.info(f"Retrieved {len(contexts)} documents")

        # Build prompt using appropriate mode
        prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.generate_prompt(
                req.query,
                contexts,
                req.mode,
                **{k: v for k, v in req.model_dump().items() if k not in ['query', 'mode', 'session_id']}
            )
        )

        if not GROQ_API_KEY:
            return ChatResponse(
                response=prompt + "\n\n*(Error: GROQ_API_KEY is missing from .env)*",
                session_id=session_id
            )

        # Call LLM with retry logic
        messages = [{"role": "user", "content": prompt}]

        # Add conversation history for context (optional - can be enabled)
        # history = conversation_store.get(session_id)[:-1]  # Exclude current query
        # if history:
        #     messages = history + messages

        max_tokens = 4096 if req.mode in ("notes", "coach", "flow") else 2048
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            call_groq_with_retry,
            messages,
            0.7,
            max_tokens,
            req.mode == "mcq"
        )

        result = data['choices'][0]['message']['content']

        # Store AI response in history
        conversation_store.add(session_id, "assistant", result)

        return ChatResponse(
            response=result,
            detected_intent=detected_intent,
            detected_mode=detected_mode,
            session_id=session_id,
            tokens_used=data.get('usage', {}).get('total_tokens')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return ChatResponse(
            response=f"**Pipeline Error:**\n```\n{str(e)}\n```",
            session_id=session_id
        )

@app.post("/api/chat/stream")
@limiter.limit("10/minute")
async def chat_stream_endpoint(request: Request, req: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            session_id = req.session_id or f"session_{id(req)}_{asyncio.get_event_loop().time()}"

            # Send session ID first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            # Auto-classify intent if mode is "auto"
            if req.mode == "auto":
                detected_intent = classify_intent_fast(req.query)
                req.mode = INTENT_TO_MODE.get(detected_intent, "tutor")
                yield f"data: {json.dumps({'type': 'intent', 'intent': detected_intent, 'mode': req.mode})}\n\n"

            # Retrieve context
            yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving context...'})}\n\n"

            contexts = await asyncio.get_event_loop().run_in_executor(
                None,
                pipeline.retrieve_intelligent,
                req.query,
                req.subject,
                3,
                req.mode
            )

            yield f"data: {json.dumps({'type': 'status', 'message': f'Retrieved {len(contexts)} documents'})}\n\n"

            # Generate prompt
            prompt = await asyncio.get_event_loop().run_in_executor(
                None,
                pipeline.generate_prompt,
                req.query,
                contexts,
                req.mode,
                **{k: v for k, v in req.model_dump().items() if k not in ['query', 'mode', 'session_id']}
            )

            if not GROQ_API_KEY:
                yield f"data: {json.dumps({'type': 'error', 'message': 'GROQ_API_KEY is missing'})}\n\n"
                return

            # Stream from Groq API
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096 if req.mode in ("notes", "coach", "flow") else 2048,
                "stream": True
            }

            if req.mode == "mcq":
                payload["response_format"] = {"type": "json_object"}

            # Make streaming request
            with requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
                stream=True
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/session/{session_id}/clear")
@limiter.limit("30/minute")
async def clear_session(request: Request, session_id: str):
    """Clear conversation history for a session."""
    conversation_store.clear(session_id)
    return {"status": "ok", "message": f"Session {session_id} cleared"}

# ============================================
# Static Files
# ============================================
os.makedirs("static", exist_ok=True)
os.makedirs("static/generated", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ============================================
# Image Download Endpoint
# ============================================
import base64
import re
from io import BytesIO

@app.post("/api/notes/download")
@limiter.limit("10/minute")
async def download_notes(request: Request, req: ChatRequest):
    """Generate downloadable notes as PNG image."""
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Get the notes content
        contexts = await asyncio.get_event_loop().run_in_executor(
            None,
            pipeline.retrieve_intelligent,
            req.query,
            req.subject,
            3,
            "notes"
        )

        prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.generate_prompt(
                req.query,
                contexts,
                "notes",
                **{k: v for k, v in req.model_dump().items() if k not in ['query', 'mode', 'session_id']}
            )
        )

        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

        # Call Groq to generate notes
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4096
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        notes_content = data['choices'][0]['message']['content']

        # Return the notes content with a flag to trigger client-side image generation
        return JSONResponse(content={
            "content": notes_content,
            "download_ready": True
        })

    except Exception as e:
        logger.exception(f"Download notes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
