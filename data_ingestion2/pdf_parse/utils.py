import pymupdf
import logging 
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def sort_boxes_reading_order(result, line_threshold=50):

    """Sort detected boxes in reading order, handling multi-column layouts"""
    # Handle both list of results (from non-batched predict) 
    # and single Results object (from batched predict loop)
    if isinstance(result, list):
        result = result[0]
        
    boxes = result.boxes
    
    elements = []
    for i in range(len(boxes.xyxy)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        cls_name = result[0].names[cls_id]
        
        elements.append({
            'index': i,
            'bbox': (x1, y1, x2, y2),
            'type': cls_name,
            'class_id': cls_id,
            'confidence': conf,
            'y_center': (y1 + y2) / 2,
            'x_center': (x1 + x2) / 2,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        })
    
    if not elements:
        return []
    
    # Get page width from image
    page_width = result[0].orig_shape[1]
    page_midpoint = page_width / 2
    
    # Separate full-width elements (titles, figures spanning both columns)
    # from column-specific elements
    full_width = []
    left_column = []
    right_column = []
    
    for elem in elements:
        elem_width = elem['x2'] - elem['x1']
        
        # If element spans more than 60% of page width, it's full-width
        if elem_width > page_width * 0.6:
            full_width.append(elem)
        # If element's center is in left half
        elif elem['x_center'] < page_midpoint:
            left_column.append(elem)
        else:
            right_column.append(elem)
    
    # Sort each group by y-position (top to bottom)
    full_width.sort(key=lambda e: e['y_center'])
    left_column.sort(key=lambda e: e['y_center'])
    right_column.sort(key=lambda e: e['y_center'])
    
    # Now interleave: full-width elements go in order,
    # then for each "row section", left column before right
    sorted_elements = []
    
    # Process by y-position zones
    all_elements = full_width + left_column + right_column
    all_elements.sort(key=lambda e: e['y1'])  # Sort by top edge
    
    # Group elements that are in the same vertical zone
    processed = set()
    
    for elem in all_elements:
        if elem['index'] in processed:
            continue
            
        elem_width = elem['x2'] - elem['x1']
        
        # Full-width element
        if elem_width > page_width * 0.6:
            sorted_elements.append(elem)
            processed.add(elem['index'])
        else:
            # Find all elements in this vertical zone
            zone_start = elem['y1']
            zone_end = elem['y2']
            
            # Extend zone to capture related elements
            zone_elements = []
            for e in all_elements:
                if e['index'] in processed:
                    continue
                e_width = e['x2'] - e['x1']
                if e_width > page_width * 0.6:
                    continue
                # Check if element overlaps with this zone
                if e['y1'] < zone_end + 50:  # Some tolerance
                    zone_elements.append(e)
                    zone_end = max(zone_end, e['y2'])
            
            # Sort zone elements: left column first, then right
            left_in_zone = [e for e in zone_elements if e['x_center'] < page_midpoint]
            right_in_zone = [e for e in zone_elements if e['x_center'] >= page_midpoint]
            
            left_in_zone.sort(key=lambda e: e['y_center'])
            right_in_zone.sort(key=lambda e: e['y_center'])
            
            for e in left_in_zone:
                if e['index'] not in processed:
                    sorted_elements.append(e)
                    processed.add(e['index'])
            
            for e in right_in_zone:
                if e['index'] not in processed:
                    sorted_elements.append(e)
                    processed.add(e['index'])
    
    return sorted_elements

def pixel_to_pdf_coordinates(bbox, img_shape, pdf_page):
    """ Convert pixel coordinates to PDF coordinates."""
    img_height, img_width = img_shape
    pdf_width, pdf_height = pdf_page.rect.width, pdf_page.rect.height

    # Scale factors
    scale_x = pdf_width / img_width
    scale_y = pdf_height / img_height

    x1, y1, x2, y2 = bbox
    
    # convert to PDF co-ordinates
    pdf_x1 = x1 * scale_x
    pdf_y1 = y1 * scale_y
    pdf_x2 = x2 * scale_x
    pdf_y2 = y2 * scale_y

    return pymupdf.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)

def extract_structured_content(page, elements, img_shape, context):
    """Orchestrator: Links captions and routes to specialized helpers."""
    enriched_elements = _associate_captions(elements)
    page_content = []
    
    for element in enriched_elements:
        # Route to specialized handlers
        handler = _get_handler(element['type'])
        chunk = handler(page, element, img_shape, context)
        
        if chunk:
            # Update heading context if we just processed a title
            if element['type'] in ['title', 'ti']:
                context['last_heading'] = chunk['content']
                
            page_content.append(chunk)
            
    return page_content

# --- Specialized Helper Functions ---

def _associate_captions(elements):
    """
    Look-ahead heuristic: Links captions to their parent element 
    based on reading order and proximity.
    """
    for i, elem in enumerate(elements):
        if 'caption' in elem['type']:
            # Search backwards for the nearest figure/table
            for j in range(i - 1, -1, -1):
                prev_elem = elements[j]
                if prev_elem['type'] in ['figure', 'table', 'isolate_formula']:
                    # Link them (we store it in metadata for the handler)
                    elem['parent_id'] = prev_elem.get('index')
                    break
    return elements

def _grid_to_markdown(grid):
    """Converts a list-of-lists (grid) into a Markdown table string."""
    if not grid or not grid[0]: return ""
    
    # 1. Format Header
    header = "| " + " | ".join(map(str, grid[0])) + " |"
    # 2. Format Separator
    separator = "| " + " | ".join(["---"] * len(grid[0])) + " |"
    # 3. Format Rows
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in grid[1:]]
    
    return "\n".join([header, separator] + rows)

def _handle_table(page, element, img_shape, context):
    """Logic for table extraction that returns Markdown directly."""
    # rect = pixel_to_pdf_coordinates(element['bbox'], img_shape, page)
    # tables = page.find_tables(clip=rect)
    
    # chunk = _create_base_chunk(element, page, rect, context)
    # if tables.tables:
    #     # CONVERT TO MARKDOWN IMMEDIATELY
    #     raw_grid = tables.tables[0].extract()
    #     chunk["content"] = _grid_to_markdown(raw_grid)
    #     chunk["format"] = "markdown_table"
    # else:
    #     chunk["content"] = "[Table detected but no cells parsed]"
    # return chunk
    return _save_visual_snippet(page, element, img_shape, context, "table")

def _handle_visual(page, element, img_shape, context):
    """Placeholder for VLM/OCR processing of figures/formulas."""
    # rect = pixel_to_pdf_coordinates(element['bbox'], img_shape, page)
    # chunk = _create_base_chunk(element, page, rect, context)
    
    # # In the future, this is where we would trigger the VLM
    # chunk["content"] = f"[{element['type'].upper()} detected at this location]"
    # chunk["is_visual"] = True
    # return chunk
    return _save_visual_snippet(page, element, img_shape, context, element['type'])

def _get_handler(element_type):
    """
    Mapping all 10 DocLayout-YOLO classes to our specialized helpers.
    """
    mapping = {
        # --- Text-based elements ---
        'title': _handle_text,
        'plain text': _handle_text,
        'figure_caption': _handle_text,
        'table_caption': _handle_text,
        'table_footnote': _handle_text,
        'formula_caption': _handle_text,

        # --- Table elements ---
        'table': _handle_table,

        # --- Visual/Complex elements ---
        'figure': _handle_visual,
        'isolate_formula': _handle_visual,

        # --- Elements to ignore ---
        'abandon': lambda *args: None  # Skips headers/footers/etc.
    }
    
    # Use .get() with a default to _handle_text for robustness
    return mapping.get(element_type, _handle_text)

def _create_base_chunk(element, page, rect, context):
    """Refactored helper to eliminate repetitive metadata code."""
    return {
        "type": element['type'],
        "page": page.number + 1,
        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
        "parent_heading": context.get('last_heading', "Document Start"),
        "confidence": element.get('confidence', 0.0)
    }

def _handle_text(page, element, img_shape, context):
    """
    Extracts text content and applies RAG-friendly formatting for 
    different text-based types.
    """
    rect = pixel_to_pdf_coordinates(element['bbox'], img_shape, page)
    # Use 'text' extraction mode for high fidelity
    raw_text = page.get_text("text", clip=rect).strip()
    
    if not raw_text:
        return None

    if element['type'] == 'title':
        # Smart Level Detection
        size = _get_element_font_size(page, rect)
        level = 1 if size > 14 else 2
        prefix = "#" * level
        formatted_content = f"{prefix} {raw_text}"
        # Update context with hierarchical info
        context['last_heading'] = raw_text 
    elif 'caption' in element['type']:
        formatted_content = f"**{element['type'].title()}:** {raw_text}"
    else:
        formatted_content = raw_text
    chunk = _create_base_chunk(element, page, rect, context)
    chunk["content"] = formatted_content
    return chunk

def _get_element_font_size(page, rect):
    """Refined helper to get the dominant font size in a region."""
    spans = page.get_text("dict", clip=rect)["blocks"]
    sizes = []
    for b in spans:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    sizes.append(s["size"])
    return max(sizes) if sizes else 10

def _save_visual_snippet(page, element, img_shape, context, element_type):
    """Saves a visual snippet to a file."""
    rect = pixel_to_pdf_coordinates(element['bbox'], img_shape, page)

    # Generate unique filename: pdfname_p1_type_index.png
    snippet_name = f"{context['pdf_name']}_p{page.number+1}_{element_type}_{element['index']}.png"
    snippet_path = os.path.join(context['snippet_dir'], snippet_name)
    
    # 1. Capture high-res pixmap (300 DPI / scale 3.0)
    pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(3, 3))
    pix.save(snippet_path)
    
    # 2. Create RAG chunk
    chunk = _create_base_chunk(element, page, rect, context)
    
    # 3. Store the Markdown image link in 'content'
    # We use a relative path 'snippets/...' so the Markdown is portable
    chunk["content"] = f"![{element_type}](snippets/{snippet_name})"
    chunk["is_visual"] = True
    chunk["local_path"] = snippet_path
    
    return chunk
