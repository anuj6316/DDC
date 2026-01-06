from zenml_flow.pipeline.visual_pipeline import layout_check_pipeline

if __name__ == "__main__":
    layout_check_pipeline.with_options(enable_cache=False)(pdf_path="/home/anuj/DDC/kb/agent0.pdf")