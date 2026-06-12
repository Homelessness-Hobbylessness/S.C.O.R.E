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

### Phase 1: Student Calibration (Building the Personal OCR Model)
At the start of a term, teachers submit three handwriting samples per student (continuous prose, numbers/symbols, exam-condition writing). These are used to create a personal OCR model either via few-shot conditioning (Approach A) or lightweight per-student fine-tuning like LoRA adapters (Approach B).

### Phase 2: Exam Ingestion
Exams arrive as scans. A pre-processing stage handles deskewing, cropping, contrast normalization, page segmentation (separating printed questions from handwritten answers), and student identification via QR code or barcode.

### Phase 3: Personalized OCR Extraction
The student's OCR profile extracts text with per-token or per-line confidence scores, linking transcribed words to bounding-box coordinates on the original scan. Low-confidence segments are flagged for the grading model or routed straight to manual review.

### Phase 4: AI Grading Engine
The grading model evaluates the extracted answer against the question, the teacher's rubric (broken into discrete, checkable criteria), and reference materials. It outputs a numerical/letter grade, a per-criterion breakdown, inline annotations, a summary comment, and a confidence score. Different subjects are routed to appropriate evaluation strategies (closed-form vs. short factual vs. extended writing).

### Phase 5: Independent Review
An independent review model (different model family, task framing, or evidence base) checks the grading model's output for internal consistency, correct rubric application, and statistical anomalies compared to the class distribution. Any issues flag the exam for closer teacher review.

### Phase 6: Teacher Verification
Teachers verify the results in a side-by-side view (original scan vs. transcribed text with grading annotations). High-confidence exams can be quickly approved, while flagged exams require closer review. All teacher edits to transcriptions or grades are logged as training signals to improve the system.

---

## Phased Rollout

- **Phase A**: Single subject, single class, OCR-only.
- **Phase B**: Add grading for closed-form subjects.
- **Phase C**: Extend to open-response subjects with one teacher closely involved in rubric design.
- **Phase D**: Multi-class, multi-teacher pilot within one school.
- **Phase E**: Multi-school deployment.

---

## Tech Stack

- **Backend**: Python, FastAPI
- **AI/ML**: TensorFlow, PyTorch, Hugging Face Transformers, Llama / Mistral / Qwen (Open-weight LLMs)
- **OCR**: TrOCR / Donut / Vision-Transformer family
- **Database**: PostgreSQL
- **Frontend**: React, TypeScript
- **Deployment**: Docker, Kubernetes (On-premise / EU data-center)
- **CI/CD**: GitHub Actions

---

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

### Running the Application

```bash
# Start with Docker Compose
docker-compose up

# Or run locally
python -m uvicorn src.api.main:app --reload
```

---

## Project Structure

```text
S.C.O.R.E/
├── docs/                 # Documentation
├── src/
│   ├── ocr/             # OCR module
│   ├── analysis/        # Error detection & analysis
│   ├── correction/      # AI-powered correction engine
│   ├── grading/         # Grading system
│   └── api/             # REST API endpoints
├── models/              # Trained ML models
├── tests/               # Unit and integration tests
└── config/              # Configuration files
```

---

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact & Support

For questions, issues, or suggestions, please reach out to the development team or open an issue on GitHub.

**Project Lead**: Urvish L. & Theodor F.  
**Repository**: https://github.com/Homelessness-Hobbylessness/S.C.O.R.E

---

**Project Start Date**: June 12, 2026  
**Expected Release Date**: 2028-2029  
**Status**: In Development
