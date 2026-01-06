import os
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class OllamaOCRProcessor:
    """
    Handles multimodal OCR enrichment using LangChain and Ollama.
    Specifically optimized for Qwen3-VL/Qwen2-VL models.
    """
    def __init__(self, model_name="qwen3-vl:235b-instruct-cloud", prompt_dir=None):
        self.model_name = model_name
        # Resolve prompt directory relative to this file if not provided
        if prompt_dir is None:
            # Assumes prompts/ is in data_ingestion2/prompts
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompt_dir = os.path.join(os.path.dirname(current_dir), "prompts")
        else:
            self.prompt_dir = prompt_dir
            
        # Initialize LangChain Ollama model
        # temperature=0 ensures deterministic results for tables/math
        self.llm = ChatOllama(model=self.model_name, temperature=0)

    def _get_base64_image(self, image_path):
        """Converts image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _load_prompt(self, element_type):
        """Loads specialized prompt from disk based on element type."""
        prompt_file = os.path.join(self.prompt_dir, f"{element_type}_prompt.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                return f.read().strip()
        
        # Default fallback if prompt file is missing
        return f"Please extract the content from this {element_type} image accurately."

    def enrich_image(self, image_path, element_type):
        """
        Main entry point: Takes an image path and type, returns VLM extracted text.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Prepare Prompt and Image
        prompt_text = self._load_prompt(element_type)
        img_base64 = self._get_base64_image(image_path)

        # 2. Construct LangChain Multimodal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ]
        )

        # 3. Invoke Model
        # Note: LangChain returns a BaseMessage, we want the .content string
        print(f"--- Sending {element_type} to Ollama ({self.model_name}) ---")
        try:
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            return f"Error during OCR enrichment: {str(e)}"

if __name__ == "__main__":
    # Quick Test logic
    # processor = OllamaOCRProcessor()
    # result = processor.enrich_image("path/to/test.png", "table")
    # print(result)
    pass
