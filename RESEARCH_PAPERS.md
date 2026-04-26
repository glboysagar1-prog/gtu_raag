# Research Papers for GTU RAG Project
## Educational RAG System - Answer Generation & XAI

---

## Overview

This document summarizes research papers relevant to building a deep, easy-to-understand RAG system for educational Q&A. The project implements:
- Hybrid retrieval (BM25 + Vector + RRF)
- LLM-based re-ranking
- Multiple answer modes (Tutor, Coach, MCQ, Notes)

---

## Topic 1: Generating Deep & Easy-to-Understand Answers

### 1. Chain-of-Thought Prompting (CoT) ⭐⭐⭐ MUST READ
**Paper:** https://arxiv.org/abs/2201.11903
**Authors:** Wei et al., 2022
**Citations:** 10,000+

**Core Idea:**
Instead of asking "What is the answer?", the model is guided to show reasoning steps first.

**Formula:**
```
Input: "Question" + "Think step by step" → Output: "Reasoning → Answer"
```

**Key Findings:**
- Significantly improves performance on arithmetic, commonsense, and symbolic reasoning
- Works by providing 2-3 step-by-step examples in the prompt
- Makes answers more explainable and traceable

**Application to Your Project:**
Add to your tutor mode prompt:
```
"Before answering, think step by step about the key concepts needed..."
```

---

### 2. Least-to-Most Prompting ⭐⭐⭐
**Paper:** https://arxiv.org/abs/2205.10625
**Authors:** Zhou et al., Google Research, 2022

**Core Idea:**
Break complex problems into simpler sub-problems solved in sequence.

**Example:**
```
Query: "Explain Operating System"
  ↓
Sub-problem 1: What is an Operating System?
Sub-problem 2: Types of OS
Sub-problem 3: Process Management
Sub-problem 4: Memory Management
  ↓
Final Answer built from all sub-problems
```

**Key Findings:**
- Standard prompting: 6% on SCAN benchmark
- Least-to-Most: 76% on SCAN benchmark
- 12x improvement for complex compositional tasks

**Application to Your Project:**
Your coach mode study plans already do this implicitly.
Could be added to tutor mode for complex topics.

---

### 3. Self-Prompting for Zero-Shot QA
**Paper:** https://arxiv.org/abs/2212.08635

**Core Idea:**
LLM generates its own QA pairs with explanations from scratch for better in-context learning.

**Application to Your Project:**
Your MCQ generator can use this to create pseudo-QA pairs from notes,
then use them to generate better questions.

---

### 4. Maieutic Prompting
**Paper:** Jung et al., 2022

**Core Idea:**
Recursive reasoning that eliminates contradictions.
Up to 20% better accuracy than CoT on Commonsense Reasoning.

**Application to Your Project:**
For ambiguous questions, LLM explores multiple possibilities,
eliminates wrong ones, then provides the most consistent answer.

---

## Topic 2: XAI - Explaining Complex AI Concepts Simply

### 5. x-[plAIn] ⭐⭐⭐ MOST PRACTICAL
**Paper:** https://arxiv.org/abs/2401.13110
**Authors:** 2024

**Core Idea:**
Custom LLM that generates clear summaries of XAI methods tailored for different audiences.

**Key Features:**
- Adapts explanations to match audience knowledge level
- Business professionals → simple analogies
- Academics → technical depth

**Application to Your Project:**
Your system already does this via:
- `marks` parameter (3 marks = brief, 10 marks = detailed)
- `mode` selection (tutor/coach/notes)

**Proposed Enhancement:**
```python
# In your prompt template
EXPLANATION_LEVEL = {
    "beginner": "Use simple language, analogies, no jargon",
    "intermediate": "Use some technical terms with explanations",
    "advanced": "Full technical depth with formulas"
}
```

---

### 6. Mind the XAI Gap
**Paper:** https://arxiv.org/abs/2506.12240 (2025)

**Core Idea:**
Two-tier explanations in one response:
1. Simple explanation for non-experts
2. Technical details for experts

**User Study Results:**
- N=56 participants
- Spearman rank correlation: 0.92 (very high accuracy)
- Improved interpretability for non-experts

**Application to Your Project:**
```
Answer structure:
┌─────────────────────────────────────┐
│ 📌 Quick Summary (for students)     │
│ In one sentence...                  │
├─────────────────────────────────────┤
│ 📖 Detailed Explanation (optional)  │
│ For those who want more depth...    │
└─────────────────────────────────────┘
```

---

### 7. LLMs for Explainable AI Survey
**Paper:** https://arxiv.org/abs/2504.00125

**Covers:**
1. Understanding user questions → appropriate explanations
2. Explaining complex ML model architectures
3. Generating counterfactual explanations

**Application to Your Project:**
For wrong MCQ answers, generate counterfactual explanations:
```
"Option B is wrong because... (if it were correct, then...)
```

---

### 8. XAI: From Inherent Explainability to LLMs
**Paper:** https://arxiv.org/abs/2501.09967 (2025)

**Comprehensive Survey covering:**
- Traditional interpretable models (linear models, decision trees)
- Modern LLM-based approaches
- High-level, semantically meaningful explanations

---

## Topic 3: RAG-Specific Research

### 9. T2-RAGBench ⭐⭐⭐ VALIDATION
**Paper:** https://arxiv.org/pdf/2604.01733

**Key Finding:**
```
Two-stage pipeline (Hybrid + Reranking) achieves:
- Recall@5: 0.816
- MRR@3: 0.605
Outperforms ALL single-stage methods
```

**Your Pipeline Matches This Pattern:**
```
Query → expand_query() → retrieve_hybrid() → rerank_documents() → LLM
```

---

### 10. Reciprocal Rank Fusion (RRF)
**Paper:** https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
**Authors:** Cormack, Clarke, Buettcher, 2009
**Citations:** 673+

**Formula:**
```
RRF_score(d) = Σ 1/(k + rank(d))  where k = 60
```

**Key Findings:**
- RRF outperforms Condorcet Fuse and CombMNZ by 4-5%
- `k=60` is optimal (tested across all values)
- No tuning required

**Your Implementation (rag_pipeline.py:584-604):**
```python
k = 60
rrf_scores = {}
for rank, doc_id in enumerate(vector_results['ids'][0]):
    rrf_scores[doc_id] += alpha * (1 / (k + rank + 1))
for rank, idx in enumerate(top_n_indices):
    rrf_scores[doc_id] += (1 - alpha) * (1 / (k + rank + 1))
```

---

## Recommended Reading Order

```
1. START HERE: Chain-of-Thought (CoT)
   ↓
2. x-[plAIn] (most practical for education)
   ↓
3. RRF Paper (validates your hybrid search)
   ↓
4. T2-RAGBench (validates your 2-stage pipeline)
   ↓
5. Least-to-Most Prompting (for complex topics)
   ↓
6. XAI Survey papers (for deeper understanding)
```

---

## Implementation Roadmap

### Phase 1: Add CoT Reasoning
```
Current: "Answer the question"
NEW: "Think step by step, then answer the question"
```

### Phase 2: Add Explanation Levels
```
"Explain at [beginner/intermediate/advanced] level"
```

### Phase 3: Add Two-Tier Explanations
```
"Provide: (1) One-sentence summary (2) Detailed explanation"
```

### Phase 4: Add Self-Prompting for MCQ
```
Generate QA pairs → Use for better MCQ creation
```

---

## References

1. Wei et al. - Chain-of-Thought Prompting: arxiv.org/abs/2201.11903
2. Zhou et al. - Least-to-Most Prompting: arxiv.org/abs/2205.10625
3. Self-Prompting: arxiv.org/abs/2212.08635
4. Maieutic Prompting: Jung et al., 2022
5. x-[plAIn]: arxiv.org/abs/2401.13110
6. Mind the XAI Gap: arxiv.org/abs/2506.12240
7. LLMs for XAI Survey: arxiv.org/abs/2504.00125
8. XAI Comprehensive: arxiv.org/abs/2501.09967
9. T2-RAGBench: arxiv.org/pdf/2604.01733
10. RRF: cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

---

*Generated for GTU RAG Project - Educational Q&A System*
*Last Updated: April 2026*