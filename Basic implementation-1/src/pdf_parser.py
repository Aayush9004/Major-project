import os
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract readable text from all pages of a PDF resume.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Combined extracted text from all pages.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a PDF or is corrupted/unreadable.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File at {pdf_path} is not a PDF file.")
        
    extracted_text_list = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                raise ValueError(f"PDF file at {pdf_path} contains no pages.")
                
            for page_idx, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text_list.append(page_text)
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise e
        raise ValueError(f"Could not read PDF file at {pdf_path}: {str(e)}")
        
    combined_text = "\n".join(extracted_text_list).strip()
    return combined_text
