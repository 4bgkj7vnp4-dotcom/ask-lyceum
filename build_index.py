
import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_PATH = "docs/handbook.pdf"
SKIP_PAGES = {1, 2, 3, 4, 39}
EMBED_MODEL = "gemini-embedding-001"

PROGRESS_FILE = "index_progress.json"
BATCH = 10


def load_pdf(path):
    reader = PdfReader(path)
    pages = []

    for number, page in enumerate(reader.pages, start=1):
        if number in SKIP_PAGES:
            continue

        text = page.extract_text() or ""
        pages.append({"page": number, "text": text})

    return pages


def split_into_chunks(pages, size=800, overlap=100):
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())
        start = 0

        while start < len(text):
            piece = text[start:start + size]

            if piece.strip():
                chunks.append({
                    "page": page["page"],
                    "text": piece
                })

            start += size - overlap

    return chunks


def embed_texts(texts):
    for attempt in range(4):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )

            return [item.values for item in response.embeddings]

        except Exception as error:
            print("Problem:", error)

            if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
                wait_time = 65
                print(f"Quota reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                time.sleep(5 * (attempt + 1))

    raise RuntimeError("Embedding failed after several tries")


# STEP 1: Read PDF
pages = load_pdf(PDF_PATH)

print("Pages kept:", len(pages))

# STEP 2: Create chunks
chunks = split_into_chunks(pages)

print("Chunks created:", len(chunks))

if not chunks:
    raise RuntimeError("No text chunks found.")

print("Example chunk (page %d):" % chunks[0]["page"])
print(chunks[0]["text"][:300])


# STEP 3: Load previous progress
if os.path.exists(PROGRESS_FILE):

    with open(PROGRESS_FILE, "r") as f:
        saved = json.load(f)

    if len(saved) != len(chunks):
        raise RuntimeError("Saved progress does not match current chunks.")

    for chunk, old_chunk in zip(chunks, saved):
        if (
            chunk["page"] == old_chunk["page"]
            and chunk["text"] == old_chunk["text"]
            and "embedding" in old_chunk
        ):
            chunk["embedding"] = old_chunk["embedding"]

    print("Previous progress loaded.")

else:
    print("Starting fresh.")


# STEP 4: Embed remaining chunks
pending = [
    i for i, chunk in enumerate(chunks)
    if "embedding" not in chunk
]

print("Already completed:", len(chunks) - len(pending))
print("Remaining:", len(pending))


for start in range(0, len(pending), BATCH):

    indices = pending[start:start + BATCH]
    batch = [chunks[i] for i in indices]

    vectors = embed_texts([c["text"] for c in batch])

    if len(vectors) != len(batch):
        raise RuntimeError("Embedding count does not match batch size.")

    for index, vector in zip(indices, vectors):
        chunks[index]["embedding"] = vector

    # Save progress after every successful batch
    with open(PROGRESS_FILE, "w") as f:
        json.dump(chunks, f)

    completed = len(chunks) - sum(
        1 for c in chunks if "embedding" not in c
    )

    print("Embedded", completed, "of", len(chunks))

    time.sleep(7)


# STEP 5: Save final index
if any("embedding" not in c for c in chunks):
    raise RuntimeError("Some chunks are missing embeddings.")

with open("index.json", "w") as f:
    json.dump(chunks, f)

print("Saved index.json with", len(chunks), "chunks")
print("Numbers per chunk:", len(chunks[0]["embedding"]))

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

print("DAY 4 COMPLETE!")
