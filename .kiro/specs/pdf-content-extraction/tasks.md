# Implementation Plan: PDF Content Extraction Pipeline

## Overview

Implementation of a comprehensive PDF content extraction system with hybrid processing methods, VLM integration, and property-based testing validation. The system processes academic papers to extract structured content in reading order for RAG applications.

## Tasks

- [x] 1. Set up project structure and core interfaces
  - Create modular package structure with clear separation of concerns
  - Define core data models and type definitions
  - Set up configuration management system
  - Initialize logging and monitoring infrastructure
  - _Requirements: 10.1, 10.4_

- [ ]* 1.1 Write property test for project structure validation
  - **Property 1: Module Import Consistency**
  - **Validates: Requirements 10.4**

- [x] 2. Implement document layout detection system
  - [x] 2.1 Create LayoutDetector class with DocLayout-YOLO integration
    - Download and initialize DocLayout-YOLO model
    - Implement element detection with configurable confidence thresholds
    - Handle 10 element types: title, plain text, figure, figure_caption, table, table_caption, table_footnote, isolate_formula, formula_caption, abandon
    - _Requirements: 1.1, 1.2, 1.4_

  - [ ]* 2.2 Write property test for element detection completeness
    - **Property 1: Element Detection Completeness**
    - **Validates: Requirements 1.1, 1.2**

  - [x] 2.3 Implement PDF to image conversion with optimization
    - Convert PDF pages to high-resolution images for processing
    - Support batch conversion for efficiency
    - Handle different PDF formats and encodings
    - _Requirements: 1.5, 8.1_

  - [ ]* 2.4 Write unit tests for PDF conversion edge cases
    - Test corrupted PDFs, password-protected files, various formats
    - _Requirements: 7.5_

- [x] 3. Implement reading order determination
  - [x] 3.1 Create ReadingOrderSorter class
    - Sort elements in natural reading order (top-to-bottom, left-to-right)
    - Handle multi-column layouts with column boundary detection
    - Process full-width elements spanning multiple columns
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ]* 3.2 Write property test for reading order consistency
    - **Property 2: Reading Order Consistency**
    - **Validates: Requirements 2.1, 2.2**

  - [x] 3.3 Implement caption-parent association logic
    - Link figure_caption, table_caption, formula_caption to parent elements
    - Use spatial proximity algorithms for association
    - Handle edge cases where captions are distant from parents
    - _Requirements: 2.3, 5.6, 6.3_

  - [ ]* 3.4 Write property test for caption-parent association
    - **Property 4: Caption-Parent Association**
    - **Validates: Requirements 2.3, 5.6**

- [x] 4. Implement text content extraction
  - [x] 4.1 Create TextExtractor class with PyMuPDF integration
    - Extract text from bounding boxes with coordinate transformation
    - Preserve formatting and paragraph structure
    - Handle Unicode and special characters correctly
    - _Requirements: 3.1, 3.2, 3.4_

  - [ ]* 4.2 Write property test for text extraction preservation
    - **Property 3: Text Extraction Preservation**
    - **Validates: Requirements 3.1, 3.2**

  - [x] 4.3 Implement text concatenation for multi-box elements
    - Handle text that spans multiple bounding boxes
    - Maintain proper spacing and line breaks
    - _Requirements: 3.3_

  - [ ]* 4.4 Write unit tests for text formatting preservation
    - Test paragraph breaks, special characters, formatting
    - _Requirements: 3.2, 3.4_

- [x] 5. Implement VLM integration for image description
  - [x] 5.1 Create VLMDescriber class with provider abstraction
    - Support local Qwen-VL as primary provider
    - Implement API-based VLM fallback (Gemini/Claude)
    - Handle authentication and rate limiting for API providers
    - _Requirements: 4.6, 10.2_

  - [x] 5.2 Implement figure description generation
    - Generate comprehensive descriptions including figure type
    - Include figure captions in VLM prompts for context
    - Extract visible text and labels from figures
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ]* 5.3 Write property test for VLM description quality
    - **Property 6: VLM Description Quality**
    - **Validates: Requirements 4.1, 4.3**

  - [x] 5.4 Implement batch processing for efficiency
    - Process multiple figures in batches when possible
    - Implement caching for similar figures
    - _Requirements: 4.5, 8.2_

  - [ ]* 5.5 Write unit tests for VLM fallback mechanisms
    - Test primary provider failure scenarios
    - _Requirements: 4.6_

- [x] 6. Implement mathematical formula processing
  - [x] 6.1 Create formula processing with VLM LaTeX conversion
    - Convert isolated formulas to LaTeX notation using VLM
    - Generate natural language representations
    - Preserve formula images for visual reference
    - _Requirements: 6.1, 6.2, 6.5_

  - [ ]* 6.2 Write property test for formula processing round-trip
    - **Property 7: Formula Processing Round-trip**
    - **Validates: Requirements 6.1, 6.2**

  - [x] 6.3 Handle both display and inline mathematical expressions
    - Distinguish between different formula types
    - Apply appropriate processing for each type
    - _Requirements: 6.4_

  - [ ]* 6.4 Write unit tests for formula type detection
    - Test display vs inline formula handling
    - _Requirements: 6.4_

- [x] 7. Implement hybrid table extraction system
  - [x] 7.1 Create TableExtractor class with multiple methods
    - Implement pdfplumber as primary extraction method
    - Add img2table as fallback for complex/scanned tables
    - Support multiple output formats: JSON, Markdown, summary
    - _Requirements: 5.1, 5.2, 5.4_

  - [ ]* 7.2 Write property test for table structure integrity
    - **Property 5: Table Structure Integrity**
    - **Validates: Requirements 5.1, 5.2**

  - [x] 7.3 Implement complex table structure handling
    - Handle merged cells and nested headers
    - Process table captions and footnotes
    - _Requirements: 5.5, 5.6_

  - [ ]* 7.4 Write unit tests for table extraction fallback
    - Test pdfplumber failure scenarios and img2table fallback
    - _Requirements: 5.4_

- [x] 8. Implement content assembly and integration
  - [x] 8.1 Create ContentAssembler class
    - Combine all processed elements in reading order
    - Generate structured output with metadata
    - Include element relationships and hierarchy
    - _Requirements: 9.1, 9.3_

  - [x] 8.2 Complete element processing integration in PDFExtractor
    - Implement _process_element method to route elements to appropriate processors
    - Handle element type-specific processing (text, figures, tables, formulas)
    - Integrate caption-parent associations
    - _Requirements: 9.1, 2.3_

  - [x] 8.3 Implement quality validation system
    - Validate reading order with confidence scoring
    - Flag incomplete content for manual review
    - Generate extraction statistics and quality metrics
    - _Requirements: 7.2, 7.3, 7.4_

  - [ ]* 8.4 Write property test for error recovery completeness
    - **Property 9: Error Recovery Completeness**
    - **Validates: Requirements 7.1, 7.2**

  - [x] 8.5 Implement graceful error handling
    - Continue processing when individual elements fail
    - Log errors with detailed context
    - Handle corrupted or malformed PDF files
    - _Requirements: 7.1, 7.5_

  - [ ]* 8.6 Write unit tests for error handling scenarios
    - Test various failure modes and recovery mechanisms
    - _Requirements: 7.1, 7.5_

- [x] 9. Implement performance optimization and parallel processing
  - [x] 9.1 Add parallel page processing capability
    - Process multiple pages concurrently when resources allow
    - Implement proper resource management and backpressure
    - _Requirements: 8.1, 8.4_

  - [ ]* 9.2 Write property test for parallel processing consistency
    - **Property 8: Parallel Processing Consistency**
    - **Validates: Requirements 8.1, 8.4**

  - [x] 9.3 Implement caching system
    - Cache layout detection results
    - Cache VLM descriptions for similar images
    - _Requirements: 8.2_

- [ ] 10. Fix critical implementation issues
  - [ ] 10.1 Fix missing ElementType import in PDFExtractor
    - Add ElementType import to core/extractor.py
    - Ensure all element type references work correctly
    - _Requirements: 1.2, 9.1_

  - [ ] 10.2 Fix VLM configuration API key handling
    - Update VLMConfig to include gemini_api_key and claude_api_key attributes
    - Fix VLMDescriber and FormulaProcessor to use correct config attributes
    - Add proper API key validation and error handling
    - _Requirements: 4.6, 10.2_

  - [ ] 10.3 Fix table extraction configuration issues
    - Update TableExtractionConfig to include fallback_methods list
    - Fix TableExtractor to handle configuration properly
    - Add proper error handling for missing extraction libraries
    - _Requirements: 5.1, 5.4_

  - [ ] 10.4 Add progress tracking and performance monitoring
    - Provide progress updates during processing
    - Track processing times and resource usage
    - _Requirements: 8.5_

  - [ ]* 10.5 Write unit tests for performance monitoring
    - Test progress tracking accuracy and performance metrics
    - _Requirements: 8.5_

- [ ] 11. Implement output formatting and integration
  - [ ] 11.1 Create multiple output format support
    - Generate JSON, Markdown, and plain text outputs
    - Ensure format consistency and completeness
    - _Requirements: 9.2, 9.4_

  - [ ]* 11.2 Write property test for output format consistency
    - **Property 10: Output Format Consistency**
    - **Validates: Requirements 9.1, 9.2**

  - [ ] 11.3 Add metadata and processing statistics
    - Include extraction timestamps and processing stats
    - Add element metadata (confidence, bounding boxes, page numbers)
    - _Requirements: 9.2, 9.5_

  - [ ]* 11.4 Write unit tests for metadata completeness
    - Test metadata accuracy and completeness
    - _Requirements: 9.2, 9.5_

- [ ] 12. Implement plugin architecture and extensibility
  - [ ] 12.1 Implement plugin architecture for custom processors
    - Allow custom element processors to be added
    - Support configuration profiles for different document types
    - _Requirements: 10.4, 10.5_

  - [ ]* 12.2 Write unit tests for configuration system
    - Test configuration loading and validation
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 13. Final integration and end-to-end testing
  - [ ] 13.1 Create end-to-end integration tests
    - Test complete document processing pipeline
    - Validate against known academic papers
    - Test various document types and layouts
    - _Requirements: All requirements_

  - [ ] 13.2 Performance benchmarking and optimization
    - Measure processing times for different document types
    - Optimize bottlenecks identified during testing
    - Validate 30-second per page performance target
    - _Requirements: 8.3_

  - [ ] 13.3 Write comprehensive property-based test suite
    - Implement all 10 correctness properties
    - Run with minimum 100 iterations each
    - Validate system behavior across diverse inputs

- [ ] 14. Final checkpoint - Ensure complete system functionality
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases