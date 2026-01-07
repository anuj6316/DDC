from concurrent.futures import ProcessPoolExecutor
from pyexpat import model
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_path
from pathlib import Path
import os
import pymupdf
import tempfile
import logging
import numpy as np
import logging
import cv2
from PIL import Image
import torch
import io
from .utils import sort_boxes_reading_order, pixel_to_pdf_coordinates, extract_structured_content

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _render_page_worker(args):
    """
    Helper function for ProcessPoolExecutor to render a single PDF page to an image.
    Args:
        args: Tuple of (pdf_path, page_index)
    Returns:
        numpy.ndarray: The rendered image as a numpy array (H, W, 3).
    """
    pdf_path, page_index = args
    try:
        # Open the PDF for this specific page render
        doc = pymupdf.open(pdf_path)
        page = doc[page_index]
        pix = page.get_pixmap()
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()
        return img_array
    except Exception as e:
        logging.error(f"Error rendering page {page_index}: {e}")
        return None

class PDFLayoutExtractor:
    def __init__(self, model_repo="juliozhao/DocLayout-YOLO-DocStructBench", model_file="doclayout_yolo_docstructbench_imgsz1024.pt"):
        logging.info(f"Loading model from {model_repo}")
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_file
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Running on: {self.device}")
        self.model = YOLOv10(model_path).to(self.device)
        logging.info("Model loaded successfully")

    def process_pdf(self, pdf_path: str, batch_size: int = 16, imgsz: int = 800) -> list:
        """
        Opens PDF, runs inference on all pages in batches, and returns structured data.
        
        Args:
            pdf_path: Path to the PDF file.
            batch_size: Number of pages to process in parallel/batch.
            imgsz: Input size for YOLO inference (default 800 for speedup).
        """
        # 1. Open Document & Initialize Context
        doc = pymupdf.open(pdf_path)
        num_pages = len(doc)
        full_document_content = []

        base_dir = os.path.dirname(pdf_path)
        snippet_dir = os.path.join(base_dir, "snippets")
        os.makedirs(snippet_dir, exist_ok=True)
        
        context = {
            'last_heading': ["Document Start"],
            "used_caption": set(),
            "snippet_dir": snippet_dir,
            "pdf_name": os.path.basename(pdf_path).replace(".pdf", "")
        }

        logging.info(f"Processing {pdf_path} ({num_pages} pages) with batch_size={batch_size}, imgsz={imgsz}")

        # 2. Iterate over pages in batches
        for i in range(0, num_pages, batch_size):
            # Collect batch indices
            current_batch_indices = list(range(i, min(i + batch_size, num_pages)))
            
            # Prepare args for worker
            worker_args = [(pdf_path, idx) for idx in current_batch_indices]
            
            batch_images = []
            
            # 3. Parallel Rendering
            with ProcessPoolExecutor() as executor:
                # Map returns results in order
                results = list(executor.map(_render_page_worker, worker_args))
                
                # Filter out failures
                batch_images = [res for res in results if res is not None]

            if not batch_images:
                continue
            
            # 4. Run Batch Inference
            # Note: doclayout-yolo/ultralytics supports list of numpy arrays
            batch_results = self.model.predict(
                batch_images,
                imgsz=imgsz,
                conf=0.25,
                device=self.device,
                verbose=False
            )

            # 5. Process Results
            for j, results in enumerate(batch_results):
                page_idx = current_batch_indices[j]
                
                # We need the open doc page object for text extraction
                page_obj = doc[page_idx]
                
                sorted_elements = sort_boxes_reading_order(results)
                
                # We need image shape (H, W) for coordinate conversion
                img_shape = batch_images[j].shape[:2]
                
                # Pass the actual PAGE OBJECT to extraction logic
                page_content = extract_structured_content(page_obj, sorted_elements, img_shape, context)
                
                full_document_content.extend(page_content)

        return full_document_content

    def pdf_to_image(self, pdf_path: str) -> list:
        """
        A generator function that yields images one by one to save ram(OOM)
        """
        doc = pymupdf.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            yield img            

    def visualize_sorting(self, pdf_path: str, batch_size: int = 15, output_path: str = None) -> str:
        """
        Returns a list of PIL Images with the reading order annotated.
        """
        img_gen = self.pdf_to_image(pdf_path)
        out_pdf = pymupdf.open() # creates a new empty pdf
        # annotated_pages = []
        finished = False

        while not finished:
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(img_gen))
                except StopIteration:
                    finished = True
                    break
            if not batch:
                break
            
            # Batch predict using our gpu/cpu
            batch_results = self.model.predict(
                batch,
                imgsz=1024,
                conf=0.25,
                device=self.device,
                verbose=False,
            )

            for j, results in enumerate(batch_results):
                ## Sort  Boxes
                sorted_elements = sort_boxes_reading_order(results)
                # this is were we will visualize the results

                draw_img = batch[j].copy()

                ## Draw Boxes and reading order Index
                for order, element in enumerate(sorted_elements):
                    x1, y1, x2, y2 = map(int, element['bbox'])
                    label = f"{order}: {element['type']}"

                    # rectangle draw
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 225), 2)

                    # draw reading order number (bg + text)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(draw_img, (x1, y1-20), (x1 + text_size[0], y1), (0,0,255), -1)
                    cv2.putText(draw_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                ## 5. Convert to PIL for zenml
                pil_img = Image.fromarray(draw_img)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                new_page = out_pdf.new_page(width=pil_img.width, height=pil_img.height)
                new_page.insert_image(new_page.rect, stream=img_byte_arr.getvalue())
                # annotated_pages.append(img_byte_arr)
            
            batch.clear()
        
        out_pdf.save(output_path)
        return output_path

if __name__ == "__main__":
    pdf_obj = PDFLayoutExtractor()
    result = pdf_obj.process_pdf("/home/anuj/DDC/kb/attention_all_you_need.pdf")
    full_md = "\n\n".join([str(c.get("content", "")) for c in result])
    with open("test.md", "r", encoding='utf-8') as f:
        f.write(full_md)