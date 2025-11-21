# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Set up the folder inside the container
WORKDIR /app

# 3. Install system tools needed for AI libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first (for caching speed)
COPY requirements.txt .

# 5. Install Python libraries
# We add --no-cache-dir to keep the file size down
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code
COPY . .

# 7. Command to run the app
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]