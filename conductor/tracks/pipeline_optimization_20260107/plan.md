# Plan: Pipeline Performance Optimization

This plan outlines the steps to optimize the DDC pipeline, focusing on asynchronous processing and caching.

## Phase 1: Asynchronous VLM Enrichment

- [ ] **Task 1: Implement Async OCR Processor**
    - Refactor `OllamaOCRProcessor` in `data_ingestion2/pdf_parse/ocr_process.py` to include an `aenrich_image` method using `ChatOllama.ainvoke`.
- [ ] **Task 2: Update ZenML Step for Concurrency**
    - Modify `enrich_markdown_step` in `zenml_flow/steps/pdf_ingestion.py` to use `asyncio.gather` for processing multiple chunks in parallel.
- [ ] **Task 3: Conductor - User Manual Verification 'Phase 1: Asynchronous VLM Enrichment' (Protocol in workflow.md)**

## Phase 2: Caching and Refinement

- [ ] **Task 4: Implement Snippet Caching**
    - Create a utility to hash image snippets and store VLM results in a local cache. Update the enrichment step to check the cache before making API calls.
- [ ] **Task 5: Refine Batch Parameters**
    - Expose `batch_size` and `imgsz` as configurable parameters in the ZenML pipeline to allow for environment-specific tuning.
- [ ] **Task 6: Conductor - User Manual Verification 'Phase 2: Caching and Refinement' (Protocol in workflow.md)**

## Phase 3: Verification and Benchmarking

- [ ] **Task 7: End-to-End Performance Test**
    - Run the optimized pipeline on a multi-page PDF and compare the execution time against previous benchmarks.
- [ ] **Task 8: Final Quality Check**
    - Ensure that the asynchronous processing does not introduce race conditions or data corruption in the final Markdown output.
- [ ] **Task 9: Conductor - User Manual Verification 'Phase 3: Verification and Benchmarking' (Protocol in workflow.md)**
