from zenml import pipeline
from zenml_flow.steps.pdf_ingestion import (
    visualize_layout_step,
    extract_content_step,
    save_markdown_step,
    enrich_markdown_step
)

@pipeline
def layout_check_pipeline(pdf_path: str):
    # 1. Extract RAG-ready content (with initial image snippets)
    extracted_data = extract_content_step(pdf_path=pdf_path)

    # 2. Enrich the visual snippets (tables/formulas) using VLM
    enriched_data = enrich_markdown_step(content=extracted_data)

    # 3. Convert enriched content to Markdown and Save
    save_markdown_step(content=enriched_data, pdf_path=pdf_path, save_local=True)
