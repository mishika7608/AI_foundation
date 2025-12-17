from docx import Document as DocxDocument
from langchain_core.documents import Document
import re
import copy

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load DOCX
doc = DocxDocument(r"D:\PythonFolder\nlp_foundation\your_file.docx")

# Extract all paragraph text
full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# Clean text
cleaned_text = clean_text(full_text)

# Create LangChain Document (single page)
documents = Document(
    page_content=cleaned_text,
    metadata={"source": "your_file.docx", "page": 1}
)

print(documents)

#CHunk Overlap: the number of characters overlapping between subsequent chunks (taking some characters from prev chunk-like to build continuity)

