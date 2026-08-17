# S.C.O.R.E. — Swift Correction & Objective Results Engine

## Overview

**S.C.O.R.E** is an advanced AI-powered exam paper analysis and grading system designed to revolutionize educational assessment. Our intelligent engine automatically detects errors, corrects mistakes, and provides objective grading using cutting-edge artificial intelligence technologies.

### Design Philosophy

Three principles run through every stage of this pipeline:
- **The human is never removed, only assisted.** Every output is treated as a draft until a teacher confirms it.
- **Confidence is a first-class citizen.** At every stage, the system produces an output plus a confidence signal.
- **Data never leaves the school's control.** Every component that touches identifiable student data runs on infrastructure the school (or a trusted regional operator) controls.

---

## Full Pipeline Specification

### Phase 1: Ingestion & Pre-Processing
Scanned PDFs are deskewed (OpenCV identifies the ruled-paper angle and rotates to horizontal), ruled lines are removed via in-painting, and a layout segmentation model separates natural-language text from mathematical equations and diagrams.

### Phase 2: Math OCR & LaTeX Extraction
Mathematical segments are converted to structured LaTeX via a dedicated math-OCR engine (GOT-OCR 2.0 in production; a multimodal LLM API is used as a substitute during prototyping since no free-tier hosting exists yet for GOT-OCR 2.0). Output is a clean Markdown document with embedded LaTeX.

### Phase 3: Tripartite Context Engine
The primary grading LLM receives three inputs simultaneously: (A) the clean OCR'd student answer, (B) the original blank question paper, and (C) the teacher's rubric (Erwartungshorizont). Using all three, it evaluates the student's logic, generates localized feedback, and proposes a score.

### Phase 4: Dual-Verification
SymPy checks generated LaTeX for syntactic validity before grading. A second, independently-prompted Auditor AI (a different model from the primary grader) cross-references the primary AI's proposed score and feedback against the rubric, flagging discrepancies such as missed valid solution paths or incorrect rubric application.

### Phase 5: Morning Review
Teachers review flagged/low-confidence items in a triage queue via a web dashboard, viewing the original handwritten snippet beside the AI's proposed score. Final grading authority always rests with the teacher; a click of "Approve" locks in a grade.

----

## Phased Rollout

- **Phase A**: Single subject, single class, OCR-only.
- **Phase B**: Add grading for closed-form subjects.
- **Phase C**: Extend to open-response subjects with one teacher closely involved in rubric design.
- **Phase D**: Multi-class, multi-teacher pilot within one school.
- **Phase E**: Multi-school deployment.

----

## Tech Stack

- **Backend**: Python, FastAPI
- **AI/ML**: TensorFlow, PyTorch, Hugging Face Transformers, Llama / Mistral / Qwen (Open-weight LLMs)
- **OCR**: GOT-OCR 2.0 (production, on-prem) / multimodal LLM API (prototyping only)
- **Database**: PostgreSQL
- **Frontend**: React, TypeScript
- **Deployment**: Docker, Kubernetes (On-premise / EU data-center)
- **CI/CD**: GitHub Actions

### Prototyping vs. Production
Prototype development uses external APIs (a multimodal LLM for OCR, one LLM for primary grading, a second distinct LLM for auditing) against synthetic/dummy exam data only; production deployment runs fully on-prem on school servers via Ollama/llama.cpp with GGUF-quantized models, with zero real student data ever leaving the school network.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Node.js 16+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/Homelessness-Hobbylessness/S.C.O.R.E.git

# Navigate to project directory
cd S.C.O.R.E

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install

# Setup environment variables
cp .env.example .env
```