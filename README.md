# OCR Core — S.C.O.R.E

[![License](https://img.shields.io/github/license/Homelessness-Hobbylessness/S.C.O.R.E)](https://github.com/Homelessness-Hobbylessness/S.C.O.R.E/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Homelessness-Hobbylessness/S.C.O.R.E/HEAD?urlpath=lab)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Homelessness-Hobbylessness/S.C.O.R.E/blob/main/demo.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Homelessness-Hobbylessness/S.C.O.R.E/main/app.py)

OCR Core is the central OCR engine and utilities used by the S.C.O.R.E demo. It provides image and PDF OCR pipelines, post-processing, evaluation utilities, and an interactive playground to try OCR quickly.

Demo 1 below — this repo contains a demo notebook and a Streamlit app to experiment with OCR models and workflows.

---

## Table of contents
- Quick interactive playground
- Features
- Quickstart (local)
- Try it (examples)
- API examples
- Configuration & models
- Evaluation & metrics
- Contributing
- License & contact

---

## Quick interactive playground

Try the project right in your browser:

- Launch a Jupyter environment (Binder):  
  https://mybinder.org/v2/gh/Homelessness-Hobbylessness/S.C.O.R.E/HEAD?urlpath=lab
- Open the demo notebook in Google Colab:  
  https://colab.research.google.com/github/Homelessness-Hobbylessness/S.C.O.R.E/blob/main/demo.ipynb
- Run the local interactive UI (Streamlit):  
  streamlit run app.py

These let you upload images/PDFs, tweak OCR settings, and view results interactively.

<details>
<summary>Interactive Playground: What you'll see</summary>

- Upload an image or PDF and get extracted text instantly.
- Toggle engines: Tesseract, EasyOCR, OCRmyPDF pipeline.
- Toggle preprocessing: binarization, deskew, denoising.
- See confidence scores and bounding boxes on the image.
- Export results as JSON, text, or searchable PDF.

</details>

---

## Features

- Unified interface for multiple OCR engines (Tesseract, EasyOCR, OCRmyPDF).
- Image preprocessing pipeline (deskew, denoise, contrast, threshold).
- PDF handling: image-extraction, OCR, searchable-PDF output.
- Post-processing: language models for punctuation, whitespace normalization.
- Confidence thresholds, per-word bounding boxes, and layout-aware extraction.
- CLI, Python API, and interactive UI (Streamlit / Notebooks).
- Evaluation scripts for WER/Character Error Rate and layout accuracy.

---

## Quickstart (local)

Prereqs:
- Python 3.8+
- pip
- Tesseract OCR installed system-wide (if using Tesseract)
  - macOS: brew install tesseract
  - Ubuntu/Debian: sudo apt install tesseract-ocr
- (Optional) Docker

Install:
