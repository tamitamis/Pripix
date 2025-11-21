# Pripix - Local AI Photo Storage


## Privacy-First. Offline.

Pripix is a self-hosted photo management tool that runs entirely on your local machine. It uses local AI models to index and caption your photos, enabling natural language search without uploading data to the cloud.


## Features

__100% Private:__ No cloud uploads. Data stays on your hard drive.

__Semantic Search:__ Search by meaning (e.g., "dog in snow") using the all-MiniLM-L6-v2 model.

__Auto-Captioning:__ Automatically describes photos upon upload using the Salesforce BLIP model.

__Background Processing:__ Uploads are instant; AI runs in the background.

__Docker Support:__ Isolated container for stable deployment.



## Installation

__Method 1: Docker (Recommended)__

__Prerequisite:__ Docker Desktop installed.

Clone/Download the repository.

__Run__:

```
docker-compose up --build
```

__Open__: `http://localhost:8000`

First run downloads ~2GB of models. Subsequent runs are instant.

__Method 2: Windows Script__

__Prerequisite:__ Python 3.10+ installed (Add to PATH).

1. Download the source code.

2. Run `start_pripix.bat`.

3. The script installs dependencies and launches the app automatically.


## Tech Stack

__Backend:__ FastAPI (Python)

__DB:__ ChromaDB (Vector Store)

__AI Models:__ BLIP (Captions), all-MiniLM-L6-v2 (Embeddings)

__Frontend:__ HTML5, JavaScript, TailwindCSS



## Directory Structure

```text pripix/
├── chroma_db/         # Local Database
├── static/
│   └── images/        # Raw image storage
├── backend.py         # Core application logic and API endpoints
├── Dockerfile         # Docker Build
├── docker-compose.yml # Docker Config
├── start_pripix.bat   # Windows Installer
└── requirements.txt   # Python dependency manifest
```


## Troubleshooting

- First run is slow: It is downloading AI models. Wait for it to finish.

- Docker "EOF" Error: Increase Docker memory to 4GB in Settings > Resources.


## Contributing

1. Fork the repository.

2. Create a feature branch (`git checkout -b feature/NewFeature`).

3. Commit changes (`git commit -m 'Implement NewFeature'`).

4. Push to the branch (`git push origin feature/NewFeature`).

5. Submit a Pull Request.

