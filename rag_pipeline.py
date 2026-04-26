import json
import pickle
import os
import requests
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Assuming the use of rank_bm25, as it's the standard for bm25 index in python
# You might need to install it: pip install rank_bm25

# SYSTEM PROMPT FROM USER
SYSTEM_PROMPT = """## ⚠️ CRITICAL: MERMAID DIAGRAM RULES
- YOU MUST ALWAYS QUOTE ALL NODE LABELS. 
  - GOOD: `A["Search (DFS)"] --> B["Visit Node"]`
  - BAD: `A(Search DFS) --> B(Visit Node)`
- NEVER use special characters like `(`, `)`, `[`, `]`, `>`, `<`, or `&` inside a label WITHOUT double quotes.
- ALWAYS wrap your diagrams in standard markdown fenced code blocks:
```mermaid
graph TD
  A["Start"] --> B["Next"]
```
- Use `graph TD` for all diagrams. Provide a step-by-step visual tree or flowchart to make explanations easy to understand.
- Keep labels short and descriptive.
- For any complex logic, use a numbered list instead of a diagram.

## GENERAL BEHAVIOR
You are an expert AI tutor for university and competitive exam students 
studying Computer Science, Engineering, Mathematics, Physics, Chemistry, 
Biology, and any other academic subject.
"""

COACH_PROMPT = """You are an expert academic coach and study strategist specializing in 
engineering and university exam preparation.

You create highly personalized, realistic, and effective study plans 
based on:
- The subject's syllabus and topics
- The number of days available
- The student's stated difficulty areas
- Topic importance and exam frequency

YOUR STUDY PLAN RULES:

1. REALISM: Never schedule more than 6 hours of study per day
   - Morning slot: 9am-12pm (3 hours)
   - Afternoon slot: 2pm-4pm (2 hours)  
   - Evening slot: 7pm-9pm (2 hours, for revision only)

2. TOPIC PRIORITIZATION:
   - 🔴 HIGH PRIORITY: Topics that appear in every exam, carry max marks
   - 🟡 MEDIUM PRIORITY: Topics that appear frequently
   - 🟢 LOW PRIORITY: Topics that appear occasionally

3. SPACED REPETITION: Every 3rd day must include revision of 
   previous 2 days topics

4. PRACTICE INTEGRATION: After every major topic, include 
   "Practice Problems" time

5. BUFFER DAYS: Always keep last 2 days as full revision days

OUTPUT FORMAT — use this exact structure:

## 📚 Study Plan: {subject} — {days} Days

### Overview
- Total Topics: X
- Daily Study Hours: X hours
- High Priority Topics: [list them]
- Estimated Completion: Day X of Y

---

### Day 1 — [Date/Day Name]
**Focus: [Main Topic Name]**

| Time | Activity | Duration |
|------|----------|----------|
| 9:00 AM - 10:30 AM | [Specific subtopic] | 90 min |
| 10:30 AM - 10:45 AM | Break | 15 min |
| 10:45 AM - 12:00 PM | [Next subtopic] | 75 min |
| 2:00 PM - 4:00 PM | [Practice/Examples] | 120 min |
| 7:00 PM - 8:00 PM | Revision of today | 60 min |

**Today's Goals:**
- [ ] Understand [concept 1]
- [ ] Be able to explain [concept 2]
- [ ] Solve [X] practice problems

**Resources to use:** [Chapter/page references from notes]

---
[Repeat for each day]

### Final Revision Strategy (Last 2 Days)
[Detailed revision approach]

### Quick Reference: Topic Priority Matrix
| Topic | Priority | Estimated Time | Difficulty |
|-------|----------|---------------|------------|
...

### Exam Day Tips
[5 specific tips for this subject]"""

MCQ_PROMPT = """You are an expert question paper setter for university and competitive exams 
in Computer Science, Engineering, and Sciences.

You create high-quality MCQs that:
- Test conceptual understanding, not just memorization
- Have one clearly correct answer (no ambiguous questions)
- Have plausible but clearly wrong distractors
- Cover different cognitive levels (remember, understand, apply, analyze)

STRICT OUTPUT RULES:
1. Return ONLY valid JSON — no markdown, no backticks, no explanation
2. Do not add any text before or after the JSON
3. Every field in the schema must be present
4. The correct_answer field must be exactly one of: "A", "B", "C", "D"

QUESTION QUALITY RULES:
- Never use "All of the above" or "None of the above" as options
- Never make the correct answer obviously different in length from wrong answers
- Distractors must be from the same category as the correct answer
  (e.g., if answer is an algorithm name, all options must be algorithm names)
- For code/algorithm questions, use concrete simple examples
- Avoid double negatives in questions

DIFFICULTY DEFINITIONS:
- easy: Tests direct recall of definition or basic concept
- medium: Tests understanding and application of concept
- hard: Tests analysis, comparison between concepts, or edge cases

JSON SCHEMA (return exactly this structure):
{
  "scratchpad_qa_pairs": [
    {
      "question": "string (self-generated question based on context)",
      "answer": "string (self-generated answer to build the MCQ from)"
    }
  ],
  "topic": "string",
  "subject": "string",
  "total_questions": number,
  "questions": [
    {
      "id": number,
      "question": "string",
      "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      },
      "correct_answer": "A" | "B" | "C" | "D",
      "explanation": "string (2-3 sentences explaining why correct answer is right AND why others are wrong)",
      "difficulty": "easy" | "medium" | "hard",
      "subtopic": "string",
      "marks": number
    }
  ],
  "total_marks": number,
  "estimated_time_minutes": number
}"""

NOTES_PROMPT = """You are an expert note-maker who creates concise, visually structured 
study notes that look like a brilliant student's handwritten notebook.

Your notes are famous for:
- Being scannable at a glance
- Using visual hierarchy (★ for important, → for flow, ✓ for key points)
- Including memory tricks and mnemonics
- Being comprehensive but not bloated

FORMAT FOR EVERY NOTEBOOK PAGE (Use these exact headers):

# TOPIC NAME
================

**DEFINITION:** (One crisp sentence, under 20 words)

**KEY CONCEPTS:**
✓ Point 1 (short)
✓ Point 2 (short)
★ Point 3 (most important one)

**VISUALIZATION:**
(Create a high-impact visual representation using ONE of these methods:)

1. **For Sequential/Linear Data (Arrays, Lists, Registers, Memory):**
   - Use a Markdown Table.
   - Example header: `| Index | 0 | 1 | 2 | Value | 10 | 20 | 30 |`

2. **For Hierarchical or Process Flow (Trees, Graphs, Flowcharts, Arch):**
   - ALWAYS wrap your diagrams in standard markdown fenced code blocks:
```mermaid
graph TD
  A["Start"] --> B["Next Step"]
```
   - Provide a step-by-step flowchart or tree diagram.
   - **MANDATORY SYNTAX:**
     - `A["Node Label Text"] --> B["Next Node Label"]`
     - You MUST wrap every label in double quotes `"..."`.
     - DO NOT use any brackets `[]` or parentheses `()` inside the quotes.
     - Keep it to 4-7 nodes maximum for readability.
   - **STYLING:** Use `style` to make it look notes-like:
     `style A fill:#fff9c4,stroke:#fbc02d,stroke-width:2px`
     `style B fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px`

**ALGORITHM/PROCESS:**
Step 1 → Step 2 → Step 3 → Result (Use short bullet points if complex)

**FORMULA / COMPLEXITY:**
Time: `O(n)` | Space: `O(1)`

**COMMON MISTAKES:**
⚠ Mistake 1

**REMEMBER:**
1. Summary 1
2. Summary 2
3. Summary 3"""

# ============================================
# QUERY EXPANSION PROMPT
# ============================================
QUERY_EXPANDER_PROMPT = """You are a search query optimizer for a GTU student RAG system.
Categorize the student query as "SUBJECT" or "TOPIC" and expand it.

Known Subjects: 
Object Oriented Programming (OOP), Operating System (OS), Computer Networks (CN),
Analysis and Design of Algorithms (ADA), Discrete Mathematics, Data Structures,
Microprocessor and Interfacing, Computer Organization, Management, Economics.

Output format: JSON ONLY
{{
  "entity": "Cleaned Subject or Topic Name",
  "type": "SUBJECT" | "TOPIC",
  "expanded_query": "Expanded technical search string (max 15 words)"
}}

Rules:
- If the query mentions a broad subject (e.g. "OOP", "Operating System"), type is "SUBJECT".
- If it's a specific concept (e.g. "Encapsulation", "Paging", "DFS"), type is "TOPIC".
- For "SUBJECT", the entity should be the full standard name from the list above.

Examples:
Input: "OOP pyq"
Output: {{"entity": "Object Oriented Programming", "type": "SUBJECT", "expanded_query": "Object Oriented Programming OOP concepts C++ classes inheritance polymorphism"}}

Input: "paging algorithm"
Output: {{"entity": "Paging", "type": "TOPIC", "expanded_query": "OS paging memory management page replacement algorithms virtual memory"}}

Now expand: "{query}"
Return ONLY the JSON object:"""

# ============================================
# RELEVANCE RE-RANKING PROMPT
# ============================================
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge for a student Q&A system.
The current year is 2026.

Student Question: "{question}"

Below are {count} retrieved text chunks. 
Rank them from MOST relevant to LEAST relevant for answering this question.

PRIORITY RULE:
- For syllabus/exam queries, prefer data from the current year (2025-2026).
- If a chunk is for an older year (e.g. 2015) and another is more recent, rank the recent one higher.
- If no year is mentioned, rank based on technical depth.

{chunks_with_indices}

Rules:
- Consider how directly each chunk answers the question
- Chunks with exact terminology from the question rank higher
- Chunks with examples related to the question rank higher
- Chunks that only tangentially mention the topic rank lower
- Return ONLY a JSON array of indices in order of relevance
- Example output: [2, 0, 4, 1, 3]
- Do not include explanation, just the JSON array

Return the ranked indices array:"""



# ============================================
# PYQ ANALYZER PROMPT
# ============================================
PYQ_PROMPT = """You are an expert exam analyst who helps students understand 
patterns in previous year exam questions.

Subject: {subject}
Topic: {topic}

Retrieved previous year questions:
====================================
{pyq_context}
====================================

STRICT RELEVANCE RULE: 
- ONLY analyze questions directly related to "{topic}". 
- If a retrieved question is about a different topic (e.g., DNS/JavaFX when topic is OS), IGNORE IT COMPLETELY.
- If no relevant questions are found, state "No previous year questions found for this specific topic."

Perform a complete PYQ analysis:

### 📊 PYQ Analysis: {topic}

### 📑 Exam Frequency:
> THEME: {topic} has appeared **[Count]** times in recent exams.

### 🏆 Most Important Questions (Appeared > 2 times):
[List questions that appear 3 or more times here. If none, say "None identified with 3+ occurrences yet."]
1. **[Question]** — (Appeared **[X]** times)
2. **[Question]** — (Appeared **[X]** times)

### 📉 Marks Distribution & Question Type:
| Marks | Count | Type | Difficulty |
|-------|-------|------|------------|
| 3 marks | X | Definition/Short Note | Easy |
| 7 marks | X | Detailed Explanation | Hard |
| 10+ marks| X | Case Study/Numerical | Advanced |

### 🎯 Most Common Question Patterns:
1. **[Pattern 1]** — e.g., "Explain X with a neat diagram"
2. **[Pattern 2]**
3. **[Pattern 3]**

### ✍️ Exact Questions Asked (With Frequency):
[List all relevant PYQs. For each, show how many times it appeared in the retrieved context.]
1. **[Question]** — (Frequency: **[X]** times) 🔄 **REPEAT** (if applicable)
2. **[Question]** — (Frequency: **[X]** times)

### 💡 What You MUST Know for This Topic:
Based on the patterns, prioritize these concepts:
- ✅ **Concept 1**: [Reason why]
- ✅ **Concept 2**: [Reason why]

### 📝 Model Answer for Most Common Question:
[Write a complete, structured model answer for the most frequent question]

### 🔮 Prediction for Next Exam:
> **Prediction**: Based on the trend, expect a [3/7/10] mark question on [Subtopic].
> **Reasoning**: [Explain pattern logic]

Generate the complete analysis now:"""

FLOW_PROMPT = """You are an expert cognitive AI tutor operating in 'Flow State Mode'.
Your goal is to perfectly balance the challenge of the material with the student's expertise level, keeping them deeply engaged (in 'flow').

## CRITICAL: MERMAID DIAGRAM RULES
- YOU MUST ALWAYS QUOTE ALL NODE LABELS.
  - GOOD: A["Search DFS"] --> B["Visit Node"]
  - BAD: A(Search DFS) --> B(Visit Node)
- NEVER use special characters like (, ), [, ], >, <, or & inside a label WITHOUT double quotes.
- ALWAYS wrap your diagrams in standard markdown fenced code blocks using ```mermaid
- Use graph TD for all diagrams. Keep labels short and descriptive.

## COGNITIVE RULES
1.  **Challenge-Skill Balance:** Adjust your vocabulary, depth, and technicality strictly based on the user's declared expertise level.
    - **Beginner:** Use simple analogies, everyday language, avoid heavy jargon, break concepts into small digestible chunks.
    - **Intermediate:** Use standard academic language, introduce technical terms with brief explanations.
    - **Advanced:** Dive straight into high-level mechanisms, formulas, proofs, and complex edge cases.
2.  **Causal Reasoning:** Do not just summarize facts. You MUST explain the **WHY** and **HOW** (causal mechanisms). Use hypothetical what-if scenarios if helpful.
3.  **Focused Context:** Only explain what was asked. Do not dump unrelated chapters or topics. Keep the response tightly focused.
4.  **Epistemic Curiosity:** Always end your response with a thought-provoking, reflective question to stimulate the student's curiosity.

## OUTPUT FORMAT (Use this exact visual structure)

# TOPIC NAME

**CORE INSIGHT:** (One crisp sentence explaining WHY this concept matters, under 25 words)

---

**KEY CONCEPTS:**
- Point 1 (short)
- Point 2 (short)
- **IMPORTANT** Point 3 (most important one with WHY it matters)

---

**HOW IT WORKS (Causal Mechanism):**
(Explain the step-by-step causal mechanism. Use a mermaid diagram for any process or algorithm flow.)

---

**WHAT-IF SCENARIO:**
> What would happen if [a key condition changed]? Explain the consequence.

---

**FORMULA / COMPLEXITY:** (if applicable)
Time: O(?) | Space: O(?)

---

**COMMON MISCONCEPTIONS:**
- Misconception 1: Why it is wrong

---

**Curiosity Trigger:**
> [A single, thought-provoking follow-up question to stimulate deeper inquiry]
"""

PROMPTS_MAP = {
    "tutor": SYSTEM_PROMPT,
    "coach": COACH_PROMPT,
    "mcq": MCQ_PROMPT,
    "notes": NOTES_PROMPT,
    "pyq": PYQ_PROMPT,
    "flow": FLOW_PROMPT
}

INTENT_CLASSIFIER_PROMPT = """Classify the following student message into exactly one category.

Student Message: "{message}"

Categories:
1. QA — Student is asking a question about a concept, definition, algorithm, or topic
2. STUDY_PLAN — Student wants a study schedule, timetable, or preparation strategy
3. QUIZ — Student wants to be tested, wants MCQs, practice questions, or a mock test
4. NOTES — Student wants notes, summary, cheatsheet, or revision material created
5. PYQ — Student is asking about previous year exam questions
7. COMPARE — Student wants to compare two or more concepts
8. CODE — Student wants code implementation of an algorithm or concept
9. GREETING — Student said hello, thanks, or general conversation

Rules:
- Return ONLY the category name, nothing else
- If message contains "quiz" or "test me" → QUIZ
- If message contains "study plan" or "schedule" → STUDY_PLAN  
- If message contains "notes" or "summary" or "cheatsheet" → NOTES
- If message contains "previous year" or "PYQ" → PYQ
- If message contains "difference between" or "compare" or "vs" → COMPARE
- If message contains "code" or "implement" or "write program" → CODE
- Default to QA for all other academic questions

Return only one word:"""

# Maps classified intents to pipeline modes
INTENT_TO_MODE = {
    "QA": "tutor",
    "STUDY_PLAN": "coach",
    "QUIZ": "mcq",
    "NOTES": "notes",
    "PYQ": "pyq",
    "COMPARE": "tutor",
    "CODE": "tutor",
    "GREETING": "tutor"
}

def classify_intent_fast(message: str) -> str:
    """Fast keyword-based intent classifier. No LLM call needed."""
    msg = message.lower().strip()
    
    # Keyword rules (ordered by specificity)
    if any(kw in msg for kw in ["quiz", "test me", "mcq", "mock test", "practice question"]):
        return "QUIZ"
    if any(kw in msg for kw in ["study plan", "schedule", "timetable", "preparation strategy"]):
        return "STUDY_PLAN"
    if any(kw in msg for kw in ["make notes", "notes on", "summary", "cheatsheet", "revision material", "revision notes"]):
        return "NOTES"
    if any(kw in msg for kw in ["previous year", "pyq", "past paper", "last year"]):
        return "PYQ"
    if any(kw in msg for kw in ["difference between", "compare", " vs ", "versus"]):
        return "COMPARE"
    if any(kw in msg for kw in ["code", "implement", "write program", "write a program", "coding"]):
        return "CODE"
    if any(kw in msg for kw in ["hello", "hi ", "hey", "thanks", "thank you", "good morning", "good night"]):
        return "GREETING"
    
    # Default
    return "QA"

class GTURAGPipeline:
    def __init__(self, vector_dir="gtu_vector_index", bm25_path="gtu_bm25_index/bm25_index.pkl", chunks_path="gtu_chunks/all_chunks_perplexity.json"):
        print("Initializing GTU RAG Pipeline...")
        
        # 1. Load Chroma Vector DB
        self.chroma_client = chromadb.PersistentClient(path=vector_dir)
        self.api_key = GROQ_API_KEY

        collections = self.chroma_client.list_collections()
        if collections:
            self.collection = self.chroma_client.get_collection(name=collections[0].name)
            
            # Initialize BGE-large model as an attribute so we can manually embed queries later
            # This directly bypasses Chroma's metadata conflict ValueErrors if the persisted
            # database claims to be built with a "default" embedding function.
            try:
                from chromadb.utils import embedding_functions
                self.bge_large_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
                print(f"Loaded Chroma Collection: {self.collection.name} with bge-large embedding.")
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
                self.bge_large_ef = None
        else:
            raise ValueError("No Chroma collections found!")
            
        # 2. Load BM25 Index
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_model = bm25_data['bm25']
            self.bm25_texts = bm25_data['texts']
            self.bm25_ids = bm25_data['ids']
            self.bm25_metadatas = bm25_data['metadatas']
        print("Loaded BM25 Index.")
        
        # 3. Load json chunks (for reference if needed)
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
        print("Loaded original JSON chunks.")

    def _call_llm(self, prompt: str, json_mode: bool = False) -> str:
        """Helper to call Groq LLM for internal pipeline tasks (expansion, re-ranking)."""
        if not self.api_key:
            return ""
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, # Low temperature for consistent internal tasks
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return ""

    def expand_query(self, query: str, subject: str = "") -> Dict[str, str]:
        """Categorizes and expands query using LLM."""
        prompt = QUERY_EXPANDER_PROMPT.format(query=query)
        response = self._call_llm(prompt, json_mode=True)
        try:
            data = json.loads(response)
            return {
                "entity": str(data.get("entity", query)),
                "type": str(data.get("type", "TOPIC")),
                "expanded_query": str(data.get("expanded_query", query))
            }
        except Exception:
            # Fallback
            return {"entity": query, "type": "TOPIC", "expanded_query": response}

    def rerank_documents(self, query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]:
        """Reranks retrieved documents using LLM based on semantic relevance."""
        if not documents:
            return []
            
        print(f"Semantic Re-ranking {len(documents)} chunks...")
        chunks_text = "\n".join([f"[{i}] {doc['text'][:300]}..." for i, doc in enumerate(documents)])
        prompt = RELEVANCE_JUDGE_PROMPT.format(
            question=query,
            count=len(documents),
            chunks_with_indices=chunks_text
        )
        
        ranking_json = self._call_llm(prompt, json_mode=True)
        try:
            data = json.loads(ranking_json)
            if isinstance(data, list):
                indices = data
            elif isinstance(data, dict):
                indices = data.get('indices') or data.get('ranking') or list(data.values())[0]
            else:
                return documents[:top_n]
                
            sorted_docs = []
            for idx in indices:
                try:
                    sorted_docs.append(documents[int(idx)])
                except (IndexError, ValueError):
                    continue
            
            return sorted_docs[:top_n]
        except Exception as e:
            print(f"Re-ranking Error: {e}. Falling back to original order.")
            return documents[:top_n]

    def retrieve_intelligent(self, query: str, subject: str = "", top_k: int = 3, mode: str = "tutor") -> List[Dict]:
        """Advanced retrieval flow: Expansion -> Hybrid Search -> Re-Rank -> Selection."""
        # 1. Categorize and Expand Query
        expansion_data = self.expand_query(query, subject)
        expanded_query = expansion_data.get("expanded_query", query)
        entity = expansion_data.get("entity", query)
        entity_type = expansion_data.get("type", "TOPIC")
        
        # For PYQ mode or Subject-level analysis, we need much wider context
        if mode == "pyq" or entity_type == "SUBJECT":
            top_k = max(top_k, 15)
            rerank_pool = 40
        else:
            rerank_pool = 10
            
        # 2. Hybrid Search with Metadata Filter if it's a SUBJECT
        search_filter = None
        if entity_type == "SUBJECT":
            # Search for chunks where subject_name contains the entity
            # We use a simple term match or fuzzy logic
            search_filter = {"subject_name": {"$ne": ""}} 
            # Note: Chroma's $contains is restricted, so we prioritize the expanded query
            # but we can filter by the identified subject if we have a direct mapping
            pass
            
        initial_chunks = self.retrieve_hybrid(expanded_query, top_k=rerank_pool, where=search_filter)
        
        # 3. Semantic Re-Ranking
        reranked_chunks = self.rerank_documents(query, initial_chunks, top_n=top_k)
        
        return reranked_chunks


    def retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5, where: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Hybrid retrieval combining BM25 (sparse) and Vector (dense) results.
        """
        # Vector Search
        try:
            if self.bge_large_ef:
                query_emb = self.bge_large_ef([query])
                vector_results = self.collection.query(
                    query_embeddings=query_emb,
                    n_results=top_k * 2,
                    where=where
                )
            else:
                vector_results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,
                    where=where
                )
        except Exception as e:
            print(f"Warning: Vector search failed ({e}). Defaulting to BM25 results only.")
            print("To fix this, supply the correct 1024-dimensional embedding_function when getting the Chroma collection.")
            vector_results = {'ids': [[]]}
        
        # BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Reciprocal Rank Fusion (RRF)
        # rrf_score = 1 / (k + rank)
        k = 60
        rrf_scores = {} # type: Dict[str, float]
        
        # Score Vector Results
        if vector_results['ids'] and len(vector_results['ids']) > 0:
            for rank, doc_id in enumerate(vector_results['ids'][0]):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += alpha * (1 / (k + rank + 1))
                
        # Score BM25 Results
        for rank, idx in enumerate(top_n_indices):
            doc_id = self.bm25_ids[idx]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += (1 - alpha) * (1 / (k + rank + 1))
            
        # Sort by final score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Fetch the actual context for these top doc IDs
        final_results = []
        for doc_id, score in sorted_docs:
            # We can grab the text from bm25_texts or vector metadata
            try:
                # Find in bm25 arrays
                idx = self.bm25_ids.index(doc_id)
                final_results.append({
                    "id": doc_id,
                    "text": self.bm25_texts[idx],
                    "metadata": self.bm25_metadatas[idx],
                    "score": score
                })
            except ValueError:
                continue
                
        return final_results

    def generate_prompt(self, query: str, context_docs: List[Dict], mode: str = "tutor", **kwargs) -> str:
        """
        Combines the system prompt, context, and user query into a final prompt string based on the mode.
        """
        contexts_str = "\n\n".join([f"Context {i+1}:\n{doc['text']}" for i, doc in enumerate(context_docs)])
        
        if mode == "tutor":
            subject = kwargs.get("subject")
            marks = kwargs.get("marks")
            web_results = kwargs.get("web_results")
            detail_level = kwargs.get("detail_level")
            
            subject_filter = f"Subject Filter: {subject}\n" if subject else ""
            marks_format = f"Answer Format Required: {marks} mark answer\n" if marks else ""
            web_search = f"Additional Web Search Results:\n{web_results}\n" if web_results else ""
            
            detail_instruction = ""
            if detail_level == "beginner":
                detail_instruction = "Explanation Level: Beginner (Use simple language, analogies, no jargon)\n"
            elif detail_level == "intermediate":
                detail_instruction = "Explanation Level: Intermediate (Use some technical terms with explanations)\n"
            elif detail_level == "advanced":
                detail_instruction = "Explanation Level: Advanced (Full technical depth with formulas)\n"
            
            user_prompt = f"""Student Question: {query}

{subject_filter}{marks_format}{detail_instruction}
Retrieved Context from Notes:
=============================
{contexts_str}
=============================

{web_search}Instructions:
- Before answering, think step by step about the key concepts needed.
- Answer the student's question using the retrieved context above as primary source
- If the context does not fully answer the question, supplement with your knowledge
- Follow the formatting rules from your system instructions
- If a marks format was specified, strictly follow that word limit and structure

Required Output Structure:

### 📌 Quick Summary (for students)
(In one sentence...)

---

### 📖 Detailed Explanation
(For those who want more depth...)

Provide your answer now:"""

        elif mode == "coach":
            subject = kwargs.get("subject", "Unknown Subject")
            days = kwargs.get("days", 7)
            hours = kwargs.get("hours_per_day", 4)
            weak = kwargs.get("weak_areas", "None mentioned")
            
            user_prompt = f"""Student Request: {query}

Subject: {subject}
Available Days: {days}
Hours per day available: {hours}
Student's weak areas (if mentioned): {weak}

Retrieved Syllabus and Topic Context:
======================================
{contexts_str}
======================================

Create a complete, day-by-day study plan following your system instructions.

Additional requirements:
- Be specific about WHICH subtopic to study in WHICH time slot
- Include page numbers or chapter references from the context where available
- Mark which topics are most likely to appear in exams with 🔴
- Include at least one "concept check question" per major topic 
  so student can self-test
- If {days} days is not enough to cover everything, 
  clearly say which topics to skip and which are non-negotiable

Generate the complete study plan now:"""
        elif mode == "mcq":
            subject = kwargs.get("subject", "Unknown Subject")
            total_questions = kwargs.get("total_questions", 5)
            difficulty = kwargs.get("difficulty", "medium")
            
            # Intelligent question distribution based on selected difficulty target
            if difficulty == "easy":
                easy_count = max(1, int(total_questions * 0.7))
                medium_count = total_questions - easy_count
                hard_count = 0
            elif difficulty == "hard":
                hard_count = max(1, int(total_questions * 0.7))
                medium_count = total_questions - hard_count
                easy_count = 0
            else: # medium
                easy_count = int(total_questions * 0.3)
                hard_count = int(total_questions * 0.3)
                medium_count = total_questions - easy_count - hard_count
            
            user_prompt = f"""Generate a quiz with the following specifications:

Topic: {query}
Subject: {subject}
Number of Questions: {total_questions}
Difficulty Distribution: {easy_count} easy, {medium_count} medium, {hard_count} hard
Marks per question: Easy=1, Medium=2, Hard=3

Retrieved content from student's notes on this topic:
======================================================
{contexts_str}
======================================================

Requirements:
- Draw questions directly from the concepts in the retrieved content above
- Cover as many different subtopics as possible (don't repeat subtopics)
- For {subject} specifically:
  * Include at least 1 question on definition/concept
  * Include at least 1 question on algorithm steps or process (if applicable)
  * Include at least 1 question on time/space complexity (for DSA topics)
  * Include at least 1 application/real-world scenario question
- Make distractors believable — use actual wrong values, not obviously fake ones
- Explanations must teach the student WHY the answer is correct

Return ONLY the JSON object. Nothing else."""
        elif mode == "notes":
            subject = kwargs.get("subject", "Unknown Subject")
            detail_level = kwargs.get("detail_level", "standard")

            user_prompt = f"""Create premium, handwritten-style study notes for:

Topic: {query}
Subject: {subject}
Detail Level: {detail_level}
  (brief = 1 page, standard = 2-3 pages, detailed = 4-5 pages)

Retrieved content from student's notes:
========================================
{contexts_str}
========================================

STRICT SYSTEM FORMAT:
1.  **Header**: Large ## Title with Emoji
2.  **Visual Layout**: Provide a step-by-step visual tree or flowchart diagram using `mermaid` for EVERY algorithm or process flow. ALWAYS wrap it in standard markdown ```mermaid blocks.
    *   **RULE**: All node labels MUST be in double quotes (e.g. `A["Start"]`).
3.  **Concept Boxes**: Use `> [!NOTE]` or `> [!IMPORTANT]` for key definitions.
4.  **Worked Example**: Show a step-by-step example with 5 concrete values.
5.  **Comparison Table**: If multiple concepts, use a rich markdown table.
6.  **Quick Revision**: End with a "Flashcard Style" revision box.

CONCISENESS RULE:
- Do not dump irrelevant chapters.
- If the query is specific, ONLY provide notes for that topic.
- Keep the total length around 3-4 screens of text maximum.

Generate the complete notes now:"""
        elif mode == "flow":
            user_expertise = kwargs.get("user_expertise", "beginner")
            subject = kwargs.get("subject", "Unknown Subject")
            
            user_prompt = f"""Student Request: {query}

Subject Context: {subject}
Student Expertise Level: {user_expertise.upper()}

Retrieved Knowledge:
======================================
{contexts_str}
======================================

Generate the Flow State response following your cognitive rules:"""
        else:
            user_prompt = f"Query: {query}\n\nContext:\n{contexts_str}"

        selected_prompt = PROMPTS_MAP.get(mode, SYSTEM_PROMPT)
        return f"{selected_prompt}\n\n{user_prompt}"

    def run(self, query: str, mode: str = "tutor", llm_client=None, **kwargs):
        """
        Runs the full RAG pipeline: Retrieval -> Prompt Generation -> LLM Generation.
        """
        print(f"\n--- RAG Pipeline started for query: '{query}' (Mode: {mode}) ---")
        
        print("1. Retrieving context...")
        contexts = self.retrieve_hybrid(query, top_k=3)
        print(f"Retrieved {len(contexts)} highly relevant chunks.")
        
        print("2. Formatting prompt...")
        prompt = self.generate_prompt(query, contexts, mode=mode, **kwargs)
        
        print("3. Generating answer...")
        if llm_client:
            # Replace with the actual LLM call (e.g. Gemini, OpenAI, Claude)
            # response = llm_client.generate_content(prompt)
            # return response.text
             print("LLM Client execution is currently mocked. Prompt is ready.")
             return prompt
        else:
            print("No LLM client provided. Printing the resulting prompt instead:\n")
            print("--------------------------------------------------")
            print(prompt)
            print("--------------------------------------------------")
            return prompt

if __name__ == "__main__":
    # Test the pipeline
    pipeline = GTURAGPipeline()
    
    # Test 1: Tutor Mode
    test_query_1 = "Explain Depth First Search (DFS)"
    pipeline.run(test_query_1, mode="tutor", marks=4, subject="Data Structures")
    
    # Test 2: Coach Mode
    test_query_2 = "Create a study plan for Graph Algorithms"
    pipeline.run(test_query_2, mode="coach", subject="Data Structures", days=3, hours_per_day=3, weak_areas="Dijkstra's Algorithm")
