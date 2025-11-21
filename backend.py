import os
import shutil
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime
import torch 

# API & Validation
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# DB & AI
import chromadb
from PIL import Image, ExifTags
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import reverse_geocoder as rg # NEW: Offline Geocoding

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
    
    # 1. Captioning Model
    caption_model_id = "Salesforce/blip-image-captioning-large"
    models["processor"] = BlipProcessor.from_pretrained(caption_model_id)
    models["caption_model"] = BlipForConditionalGeneration.from_pretrained(caption_model_id).to(DEVICE)
    
    # 2. Embedding Model (MiniLM)
    print("--- Loading MiniLM... ---")
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
    date_taken: Optional[str] = None # NEW
    location: Optional[str] = None   # NEW

# --- HELPERS (EXIF & LOGIC) ---

def get_decimal_from_dms(dms, ref):
    """Helper to convert GPS degrees/minutes/seconds to decimal"""
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_metadata(image_path):
    """
    Extracts Date and Location (City, Country) from Image EXIF.
    Returns: (date_string, location_string)
    """
    date_str = "Unknown Date"
    location_str = "Unknown Location"
    
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return date_str, location_str

        # Map Exif tags to names
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in ExifTags.TAGS
        }

        # 1. Extract Date
        if 'DateTimeOriginal' in exif:
            # Format: YYYY:MM:DD HH:MM:SS
            dt_obj = datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
            # Convert to nice string: "15 October 2023"
            date_str = dt_obj.strftime("%d %B %Y")
        
        # 2. Extract GPS
        if 'GPSInfo' in exif:
            gps_info = exif['GPSInfo']
            
            # We need Lat, LatRef, Lon, LonRef
            if 2 in gps_info and 4 in gps_info:
                lat = get_decimal_from_dms(gps_info[2], gps_info[1])
                lon = get_decimal_from_dms(gps_info[4], gps_info[3])
                
                # Offline Reverse Geocode
                results = rg.search((lat, lon)) # Returns list of dicts
                if results:
                    city = results[0].get('name', '')
                    country = results[0].get('cc', '') # Country code like 'US' or 'IN'
                    location_str = f"{city}, {country}"

    except Exception as e:
        print(f"Metadata Error: {e}")

    return date_str, location_str

def generate_caption(image_path: str):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = models["processor"](raw_image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out = models["caption_model"].generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=30,
            num_beams=3,
            repetition_penalty=1.2
        )
    return models["processor"].decode(out[0], skip_special_tokens=True)

def get_embedding(text: str):
    return models["embedder"].encode(text).tolist()

def process_image_background(file_id: str, filename: str, file_path: str):
    try:
        print(f"Background: Processing {filename}...")
        
        # 1. Extract Real Metadata
        date_taken, location = extract_metadata(file_path)
        
        # 2. Generate AI Caption
        ai_caption = generate_caption(file_path)
        
        # 3. CREATE HYBRID CONTEXT
        # We fuse the metadata INTO the caption for the Search Engine
        # e.g. "A dog running. Photo taken in London, GB on 12 October 2023."
        full_context_caption = f"{ai_caption}. Photo taken in {location} on {date_taken}."
        
        real_vector = get_embedding(full_context_caption)
        
        collection.update(
            ids=[file_id],
            embeddings=[real_vector],
            metadatas=[{
                "filename": filename, 
                "caption": ai_caption, # We show just the AI caption to user
                "path": file_path,
                "date": date_taken,    # Store separately for UI
                "location": location   # Store separately for UI
            }]
        )
        print(f"Finished {filename}: {full_context_caption}")
        
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
            
        placeholder_caption = "Processing..."
        placeholder_vector = get_embedding("pending") 

        collection.add(
            ids=[file_id],
            embeddings=[placeholder_vector],
            metadatas=[{
                "filename": filename, 
                "caption": placeholder_caption, 
                "path": file_path,
                "date": "",
                "location": ""
            }]
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
                url=f"/static/images/{meta['filename']}",
                date_taken=meta.get('date', 'Unknown'),
                location=meta.get('location', 'Unknown')
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
            url=f"/static/images/{meta['filename']}",
            date_taken=meta.get('date', 'Unknown'),
            location=meta.get('location', 'Unknown')
        ))
    return response

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

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
        raise HTTPException(status_code=500, detail="Failed to delete photo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)