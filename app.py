from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
import openai
import os
from endee import Client

app = FastAPI()

# Connect to Endee (make sure Endee server runs on port 8080)
client = Client(host="localhost", port=8080)
collection = client.get_or_create_collection("rag_docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Endee Powered OCR RAG Chatbot</h2>
    <form action="/ask/" method="post" enctype="multipart/form-data">
        Upload Image: <input type="file" name="file"><br><br>
        Ask Question: <input type="text" name="question"><br><br>
        <input type="submit">
    </form>
    """


@app.post("/ask/")
async def ask(file: UploadFile = File(...), question: str = Form(...)):

    file_path = file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # OCR
    text = pytesseract.image_to_string(Image.open(file_path))

    # Chunk text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Create embeddings
    embeddings = embedder.encode(chunks)

    # Store embeddings in Endee
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            id=str(i),
            vector=embedding.tolist(),
            metadata={"text": chunk}
        )

    # Embed question
    query_vector = embedder.encode(question)

    # Retrieve from Endee
    results = collection.search(
        vector=query_vector.tolist(),
        top_k=3
    )

    context = "\n\n".join([r.metadata["text"] for r in results])

    prompt = f"""
    Answer using only the context below:

    {context}

    Question: {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response.choices[0].message["content"]}
