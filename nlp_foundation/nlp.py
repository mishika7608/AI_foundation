from pypdf import PdfReader #lightweight pdf reader 
from langchain_core.documents import Document #imports standard container
import copy
import re

#loader_pdf = PyPDFLoader(r"D:\PythonFolder\nlp_foundation\Introduction_to_Data_and_Data_Science.pdf")
# pages_pdf = loader_pdf.load()
# print(pages_pdf)

reader = PdfReader(r"D:\PythonFolder\nlp_foundation\Introduction_to_Data_and_Data_Science.pdf")
documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() #extract raw text, and none if empty or image
    if text:
        documents.append(
            Document( #craetes langchain compatible document object 
                page_content=text, #used for chunking, embedding, RAG
                metadata={"page":i} #source citation, page preferences, debugging
            )
        )
print(documents)
raw_documents = documents
cleaned_documents = copy.deepcopy(raw_documents)


def clean_text(text: str) -> str:
    # Remove excessive whitespace (newlines, tabs, etc.)
    text = re.sub(r"\s+", " ", text)

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)

    return text.strip()

for doc in cleaned_documents:
    doc.page_content = clean_text(doc.page_content)

print(raw_documents[0].page_content[:200])
print(cleaned_documents[0].page_content[:200])