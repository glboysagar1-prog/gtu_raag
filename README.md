# GTU RAG - Educational Q&A System

A sophisticated Retrieval-Augmented Generation (RAG) system built for GTU (Gujarat Technological University) students. Provides intelligent answers, study plans, quizzes, and handwritten-style notes using hybrid search (BM25 + Vector) with LLM-based query expansion and re-ranking.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### 5 Operational Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Auto** | Intent classification → auto-routing | Natural language queries |
| **Tutor** | Q&A with mark-based formatting | Exam preparation (3/4/7/10 marks) |
| **Coach** | Study plan generator | Creating day-by-day schedules |
| **MCQ** | Quiz generator | Self-assessment and practice |
| **Notes** | Handwritten-style notes | Revision material with diagrams |

### Technical Highlights

- **Hybrid Retrieval**: BM25 (sparse) + ChromaDB vector search (dense) with Reciprocal Rank Fusion (RRF)
- **Query Expansion**: LLM optimizes search queries for better retrieval
- **Semantic Re-ranking**: LLM as relevance judge for context selection
- **Intent Classification**: Keyword-based classifier with LLM fallback
- **Mermaid Diagrams**: Automatic visualization of algorithms and processes
- **Streaming Responses**: Real-time token streaming for better UX

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or uv
- Groq API key (free at https://console.groq.com/keys)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gtu-rag.git
cd gtu-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

5. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. **Open in browser**
```
http://localhost:8000
```

## API Endpoints

### POST /api/chat
Main chat endpoint for all modes.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain Depth First Search",
    "mode": "tutor",
    "marks": 4,
    "subject": "Data Structures"
  }'
```

### GET /api/health
Health check endpoint.

```bash
curl http://localhost:8000/api/health
```

### GET /api/chat/stream
Streaming endpoint with Server-Sent Events.

```bash
curl -N http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain OS",
    "mode": "tutor"
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `GROQ_MODEL` | Model to use | `llama-3.1-8b-instant` |
| `HANDWRITTEN_ANSWER` | Enable handwritten notes style | `true` |

### Mode Parameters

**Tutor Mode:**
- `marks`: 3, 4, 7, or 10
- `detail_level`: beginner, intermediate, advanced

**Coach Mode:**
- `days`: Number of days for study plan
- `hours_per_day`: Daily study hours
- `weak_areas`: Comma-separated weak topics

**MCQ Mode:**
- `total_questions`: Number of questions (1-15)
- `difficulty`: easy, medium, hard

**Notes Mode:**
- `detail_level`: brief, standard, detailed

## Project Structure

```
gtu-rag/
├── main.py                 # FastAPI application entry point
├── rag_pipeline.py         # Core RAG pipeline implementation
├── verify_index.py         # Index verification utility
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Modern Python packaging
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── RESEARCH_PAPERS.md      # Research documentation
├── static/
│   ├── index.html          # Frontend UI
│   ├── script.js           # Frontend logic
│   └── style.css           # Styling
├── gtu_chunks/             # Document chunks data
├── gtu_bm25_index/         # BM25 sparse index
├── gtu_vector_index/       # ChromaDB vector index
└── tests/
    ├── test_pipeline.py    # Pipeline unit tests
    ├── test_hybrid_search.py  # Hybrid search tests
    └── test_intent_classifier.py  # Intent classifier tests
```

## Docker Support

### Build and Run with Docker

```bash
# Build the image
docker build -t gtu-rag .

# Run the container
docker run -p 8000:8000 --env-file .env gtu-rag
```

### Docker Compose

```bash
docker-compose up --build
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_pipeline.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Static UI)                      │
│  5-mode toggle, Mermaid diagrams, Markdown rendering         │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                       │
│  Rate limiting, Request validation, Logging middleware       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 RAG PIPELINE (GTURAGPipeline)                │
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌──────────────────────┐   │
│  │  Query   │ → │  Query    │ → │ Hybrid Retrieval     │   │
│  │ Expansion│   │ Expansion │   │ (BM25 + Vector + RRF)│   │
│  │ (LLM)    │   │ (JSON)    │   │                      │   │
│  └──────────┘   └───────────┘   └──────────────────────┘   │
│                                              │               │
│                                              ▼               │
│  ┌──────────┐   ┌───────────┐   ┌──────────────────────┐   │
│  │ LLM      │ ← │ Prompt    │ ← │ Re-Ranking           │   │
│  │ Generation│  │ Generation│   │ (LLM Relevance Judge)│   │
│  │ (Groq)   │   │ (Mode-based)│ │                      │   │
│  └──────────┘   └───────────┘   └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  ChromaDB (BGE-large embeddings) + BM25 Index               │
└─────────────────────────────────────────────────────────────┘
```

## Research-Backed Features

This implementation incorporates findings from academic research:

1. **Chain-of-Thought Prompting** (Wei et al., 2022) - Step-by-step reasoning for better answers
2. **Reciprocal Rank Fusion** (Cormack et al., 2009) - Optimal hybrid search combination
3. **Two-Tier Explanations** - Quick summary + detailed explanation structure
4. **Explanation Levels** (x-[plAIn] paper) - Audience-adapted responses

See [RESEARCH_PAPERS.md](RESEARCH_PAPERS.md) for detailed documentation.

## Troubleshooting

### Common Issues

**"GROQ_API_KEY is not set"**
- Copy `.env.example` to `.env` and add your API key

**"No Chroma collections found"**
- Ensure `gtu_vector_index/` directory exists with valid index files

**"Vector search failed"**
- Check that `sentence-transformers` is installed correctly
- Verify BGE-large model can be downloaded

**"Failed to parse MCQ JSON"**
- This is an LLM output issue - try regenerating or use a different model

### Performance Tips

- Use `--reload` only in development
- For production: `uvicorn main:app --workers 4`
- Enable response caching for repeated queries
- Consider Redis for session storage at scale

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Groq for fast LLM inference
- ChromaDB for vector storage
- SentenceTransformers for embeddings
- GTU for course material

## Contact

For questions or support, please open an issue on GitHub.
