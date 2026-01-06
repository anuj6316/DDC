from zenml import step
from typing import List
from PIL import Image
import os
import logging
from data_ingestion2.pdf_parse.ocr_process import OllamaOCRProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def visualize_layout_step(pdf_path: str) -> str:
    """
    Ingests a PDF, detects layout, and returns images annotated with 
    the sorted reading order for manual verification.
    """
    # 1. Initialize logic
    analyzer = PDFLayoutExtractor()
    
    output_path = pdf_path.replace(".pdf", "_annotated.pdf")

    # 2. Get annotated images
    # We use the method we just created
    saved_path = analyzer.visualize_sorting(pdf_path, output_path=output_path)
    # 3. Return list. ZenML will display these in the dashboard.
    return f"Full Visualized Layout saved at {saved_path}"
    # return Image.open(saved_path)

@step
def save_markdown_step(content: List[dict], pdf_path: str, save_local: bool = True) -> str:
    """
    ZenML step with a toggle to save locally or just return the string.
    """
    # Simply join the pre-formatted content strings from each chunk
    full_md = "\n\n".join([str(c.get('content', '')) for c in content])
    
    if save_local:
        output_path = pdf_path.replace(".pdf", "_extracted.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_md)
        logging.info(f"Markdown saved locally to: {output_path}")

    return full_md

@step
def extract_content_step(pdf_path: str) -> List[dict]:
    """
    ZenML step: Extract RAG-ready structured content
    """
    from data_ingestion2.pdf_parse.doclayout import PDFLayoutExtractor
    analyzer = PDFLayoutExtractor()
    content = analyzer.process_pdf(pdf_path)
    logging.info(f"Extracted complete. Found {len(content)} elements")

    return content

@step
def enrich_markdown_step(content: List[dict], model_name: str = "qwen3-vl:235b-instruct-cloud") -> List[dict]:
    """
    ZenML step: Enrich visual elements (tables, formulas, figures) using a VLM via Ollama.
    """
    processor = OllamaOCRProcessor(model_name=model_name)
    enriched_content = []
    
    for chunk in content:
        # Check if the chunk is visual and has a path to the image snippet
        if chunk.get("is_visual") and "local_path" in chunk:
            element_type = chunk.get("type", "visual")
            image_path = chunk["local_path"]
            
            logging.info(f"Enriching {element_type} snippet using VLM: {image_path}")
            vlm_text = processor.enrich_image(image_path, element_type)
            
            # Format the content to include the VLM text and the original snippet reference
            # Use relative path for portability in the dashboard/markdown
            snippet_name = os.path.basename(image_path)
            rel_path = f"snippets/{snippet_name}"
            
            chunk["content"] = (
                f"### VLM Enrichment ({element_type})\n"
                f"{vlm_text}\n\n"
                f"---\n"
                f"*(Original Snippet: ![{element_type}]({rel_path}))*"
            )
        
        enriched_content.append(chunk)
    
    return enriched_content