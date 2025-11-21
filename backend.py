import os
import shutil
import uuid
from typing import List
from contextlib import asynccontextmanager
import torch 

# API & Validation
# ADDED: FileResponse is needed for the new download feature
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# DB & AI
import chromadb
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
UPLOAD_DIR = "static/images"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pripix_photos"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- GLOBAL STATE ---
models = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"--- PRIPX: Loading AI Models on {DEVICE.upper()}... ---")
    
    model_id = "Salesforce/blip-image-captioning-large"
    
    models["processor"] = BlipProcessor.from_pretrained(model_id)
    models["caption_model"] = BlipForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    models["embedder"] = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    print("--- PRIPX: Models Loaded. System Ready. ---")
    yield
    models.clear()

app = FastAPI(title="Pripix API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- DATABASE ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- MODELS ---
class SearchResult(BaseModel):
    id: str
    filename: str
    caption: str
    score: float
    url: str

# --- HELPERS ---
def generate_caption(image_path: str):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = models["processor"](raw_image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out = models["caption_model"].generate(
            **inputs,
            max_new_tokens=70,
            min_new_tokens=20,
            num_beams=3,
            repetition_penalty=1.1
        )
    return models["processor"].decode(out[0], skip_special_tokens=True)

def get_embedding(text: str):
    return models["embedder"].encode(text).tolist()

def process_image_background(file_id: str, filename: str, file_path: str):
    try:
        print(f"Background: Processing {filename}...")
        real_caption = generate_caption(file_path)
        real_vector = get_embedding(real_caption)
        
        collection.update(
            ids=[file_id],
            embeddings=[real_vector],
            metadatas=[{"filename": filename, "caption": real_caption, "path": file_path}]
        )
        print(f"Background: Finished {filename}. Caption: {real_caption}")
        
    except Exception as e:
        print(f"Background Error for {filename}: {e}")

# --- ENDPOINTS ---

@app.post("/upload/")
async def upload_photo(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        extension = file.filename.split(".")[-1]
        filename = f"{file_id}.{extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        placeholder_caption = "Processing AI details..."
        placeholder_vector = get_embedding("processing pending") 

        collection.add(
            ids=[file_id],
            embeddings=[placeholder_vector],
            metadatas=[{"filename": filename, "caption": placeholder_caption, "path": file_path}]
        )

        background_tasks.add_task(process_image_background, file_id, filename, file_path)

        return {
            "status": "queued", 
            "id": file_id, 
            "caption": placeholder_caption, 
            "url": f"/static/images/{filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/", response_model=List[SearchResult])
async def search_photos(q: str, limit: int = 10):
    query_vector = get_embedding(q)
    results = collection.query(query_embeddings=[query_vector], n_results=limit)
    
    response = []
    if results['ids']:
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            response.append(SearchResult(
                id=results['ids'][0][i],
                filename=meta['filename'],
                caption=meta['caption'],
                score=results['distances'][0][i],
                url=f"/static/images/{meta['filename']}"
            ))
    return response

@app.get("/gallery")
async def get_all_photos(limit: int = 20):
    count = collection.count()
    if count == 0: return []
    real_limit = min(count, limit)
    data = collection.peek(limit=real_limit)
    
    response = []
    for i in range(len(data['ids'])):
        meta = data['metadatas'][i]
        response.append(SearchResult(
            id=data['ids'][i],
            filename=meta['filename'],
            caption=meta['caption'],
            score=0.0,
            url=f"/static/images/{meta['filename']}"
        ))
    return response

# NEW: Specific endpoint to force file download
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # 'attachment' tells the browser to download, not open
    return FileResponse(
        path=file_path, 
        filename=filename, 
        media_type='application/octet-stream'
    )

@app.delete("/photos/{photo_id}")
async def delete_photo(photo_id: str):
    try:
        data = collection.get(ids=[photo_id])
        if not data['ids']:
            raise HTTPException(status_code=404, detail="Photo not found")
        file_path = data['metadatas'][0]['path']
        if os.path.exists(file_path):
            os.remove(file_path)
        collection.delete(ids=[photo_id])
        return {"status": "deleted", "id": photo_id}
    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete photo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)