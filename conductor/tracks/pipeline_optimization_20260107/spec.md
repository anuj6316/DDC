# Specification: Pipeline Performance Optimization

## Overview
This track focuses on reducing the end-to-end processing time of the DDC pipeline. The primary bottlenecks identified are the sequential layout analysis and the synchronous waiting for VLM enrichment results.

## Objectives
- Implement asynchronous VLM enrichment to process multiple image snippets in parallel.
- Introduce a caching layer to skip VLM enrichment for previously processed image snippets.
- Refine the batch processing logic for layout detection (YOLO).
- Optimize the snippet generation and storage process.

## Key Components

### 1. Asynchronous VLM Enrichment
- **Goal:** Shift from sequential `invoke` to concurrent `ainvoke` using `asyncio.gather`.
- **Target:** `data_ingestion2/pdf_parse/ocr_process.py` and `zenml_flow/steps/pdf_ingestion.py`.

### 2. Enrichment Caching
- **Goal:** Create a content-addressable cache (e.g., using SHA-256 hashes of image files).
- **Location:** Local file system or a lightweight database.

### 3. Layout Detection Refinement
- **Goal:** Fine-tune batch sizes and resolution (imgsz) based on hardware capabilities.
- **Target:** `data_ingestion2/pdf_parse/doclayout.py`.

## Success Criteria
- End-to-end processing time reduced by at least 50% for standard documents.
- No regression in extraction accuracy or layout reconstruction.
- Verified parallelization of VLM requests in logs.
