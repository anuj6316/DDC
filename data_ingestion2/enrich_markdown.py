import os
import re
from pdf_parse.ocr_process import OllamaOCRProcessor

def enrich_markdown(md_path, model_name="qwen3-vl:235b-instruct-cloud"):
    """
    Scans a markdown file for image snippets and replaces them with 
    text extracted by the OllamaOCRProcessor.
    """
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found at {md_path}")
        return

    # 1. Initialize our new LangChain-based processor
    processor = OllamaOCRProcessor(model_name=model_name)
    
    with open(md_path, "r") as f:
        content = f.read()

    # Regex to find: ![element_type](snippets/filename.png)
    # Group 1: element_type
    # Group 2: relative_path
    pattern = r"!\[(.*?)\]\((snippets/.*?\.png)\)"
    matches = re.findall(pattern, content)

    if not matches:
        print("No image snippets found for enrichment.")
        return

    print(f"Found {len(matches)} snippets to enrich. Starting processing...")

    for element_type, rel_path in matches:
        # Resolve absolute path for the processor
        abs_path = os.path.join(os.path.dirname(md_path), rel_path)
        
        if os.path.exists(abs_path):
            # 2. Get the VLM extraction for this specific snippet
            enriched_content = processor.enrich_image(abs_path, element_type)
            
            # 3. Create the RAG-friendly replacement block
            # We keep the original image link in a comment or small note for verification
            replacement = (
                f"<!-- START VLM ENRICHMENT ({element_type}) -->\n"
                f"{enriched_content}\n\n"
                f"*(Source: ![{element_type}]({rel_path}))*\n"
                f"<!-- END VLM ENRICHMENT -->"
            )
            
            # Use 1 to replace only the first occurrence in each loop iteration
            target = f"![{element_type}]({rel_path})"
            content = content.replace(target, replacement, 1)
            print(f"✅ Enriched: {rel_path} ({element_type})")
        else:
            print(f"⚠️ Snippet file not found: {abs_path}")

    # 4. Save the final enriched file
    output_path = md_path.replace(".md", "_enriched.md")
    with open(output_path, "w") as f:
        f.write(content)
        
    print("-" * 30)
    print(f"DONE! Enriched file saved to: {output_path}")

if __name__ == "__main__":
    # Point this to your generated markdown file
    PATH_TO_MD = "/home/anuj/DDC/kb/agent0_extracted.md"
    enrich_markdown(PATH_TO_MD)
