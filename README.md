Pripix - Local AI Photo Storage

Privacy-First. Offline. Intelligent.

Pripix is a self-hosted photo management tool that runs entirely on your local machine. It uses local AI models to index and caption your photos, enabling natural language search without uploading data to the cloud.

Features

100% Private: No cloud uploads. Data stays on your hard drive.

Semantic Search: Search by meaning (e.g., "dog in snow") using the all-MiniLM-L6-v2 model.

Auto-Captioning: Automatically describes photos upon upload using the Salesforce BLIP model.

Background Processing: Uploads are instant; AI runs in the background.

Docker Support: Isolated container for stable deployment.

Installation

Method 1: Docker (Recommended)

Prerequisite: Docker Desktop installed.

Clone/Download the repository.

Run:

docker-compose up --build


Open: http://localhost:8000

First run downloads ~2GB of models. Subsequent runs are instant.

Method 2: Windows Script

Prerequisite: Python 3.10+ installed (Add to PATH).

Download the source code.

Run start_pripix.bat.

The script installs dependencies and launches the app automatically.

Tech Stack

Backend: FastAPI (Python)

DB: ChromaDB (Vector Store)

AI Models: BLIP (Captions), all-MiniLM-L6-v2 (Embeddings)

Frontend: HTML5, JavaScript, TailwindCSS

Directory Structure

pripix/
├── chroma_db/         # Local Database
├── static/images/     # Raw Photos
├── backend.py         # App Logic
├── Dockerfile         # Docker Build
├── docker-compose.yml # Docker Config
├── start_pripix.bat   # Windows Installer
└── requirements.txt   # Python Libs


Troubleshooting

First run is slow: It is downloading AI models. Wait for it to finish.

Docker "EOF" Error: Increase Docker memory to 4GB in Settings > Resources.
