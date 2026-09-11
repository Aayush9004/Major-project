import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_resume_pdf():
    os.makedirs("data/resumes", exist_ok=True)
    pdf_path = "data/resumes/sample_candidate.pdf"
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "John Doe")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 730, "Software Developer")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 700, "Experience:")
    c.setFont("Helvetica", 11)
    c.drawString(100, 680, "3 years of experience in software development.")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 650, "Education:")
    c.setFont("Helvetica", 11)
    c.drawString(100, 630, "B.Tech in Computer Engineering.")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 600, "Skills:")
    c.setFont("Helvetica", 11)
    c.drawString(100, 580, "Python, SQL, FastAPI, Machine Learning, React, Git.")
    
    c.save()
    print(f"Sample PDF resume generated at: {pdf_path}")

if __name__ == "__main__":
    create_sample_resume_pdf()
