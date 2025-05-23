import sys
from docx import Document

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\\n'.join(full_text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_docx_text.py <path_to_docx>")
        sys.exit(1)
    docx_path = sys.argv[1]
    text = extract_text_from_docx(docx_path)
    print(text)
