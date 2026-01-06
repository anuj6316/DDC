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

    def process_pdf(self, pdf_path: str) -> list:
            """
            Opens PDF, runs inference on all pages, and returns structured data.
            """
            doc = pymupdf.open(pdf_path)
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
            for i, page in enumerate(doc):
                # 1. Convert Page to Image for YOLO
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

                # 2. Predict
                # Note: Moved configuration params here or pass them in __init__
                results = self.model.predict(
                    img_array,
                    imgsz=1024,
                    conf=0.25,
                    verbose=False
                )

                # 3. Sort Boxes (using your utils)
                sorted_elements = sort_boxes_reading_order(results)

                # 4. Extract Content (using your utils)
                # We explicitly pass the image shape so coordinate conversion works
                img_shape = (pix.height, pix.width)
                page_content = extract_structured_content(page, sorted_elements, img_shape, context)

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
    
    def process_pdf_optimized(self, pdf_path: str, batch_size: int = 15) -> list:
        all_images = self.pdf_to_image(pdf_path)
        for i in range(len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            batch_results = self.model.predict(
                batch,
                imgsz=1024,
                conf=0.25,
            )

            for j, results in enumerate(batch_results):
                page_idx = i+j
                page = doc[page_idx]

                sorted_elements = sort_boxes_reading_order(results)
                image_shape = batch[j].shape[:2]
        pass            

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
