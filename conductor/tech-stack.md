# Technology Stack

## Core Language & Runtime
- **Language:** Python 3.12+
- **Package Manager:** `uv` (Fast Python package installer and resolver)
- **Task Runner:** `poethepoet` (For managing project-level tasks and automation)

## Orchestration & MLOps
- **Orchestration:** ZenML (Reproducible pipeline management and artifact tracking)

## Document Processing & Computer Vision
- **Layout Analysis:** `DocLayout-YOLO` (YOLOv10-based high-precision layout detection)
- **PDF Engine:** `PyMuPDF` (fitz) (High-fidelity text and image extraction)
- **Image Processing:** `OpenCV`, `Pillow` (PIL), `NumPy`
- **PDF-to-Image:** `pdf2image`

## Vision-Language Models (VLM) & OCR
- **VLM Framework:** `LangChain` (LLM/VLM orchestration and abstraction)
- **Local Inference:** `Ollama` (Running Qwen2-VL / Qwen2.5-VL models locally)
- **Additional OCR:** `img2table`, `PaddleOCR` (as fallback/utility)

## Environment & Configuration
- **Version Control:** Git
- **Configuration:** `pyproject.toml`
