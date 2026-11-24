from pypdf import PdfReader

reader = PdfReader("docs/handbook.pdf")
print("Total pages:", len(reader.pages))

text = reader.pages[9].extract_text()
print("Characters on page 10:", len(text))
print(text[:500])