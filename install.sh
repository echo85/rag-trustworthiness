#!/bin/bash
apt-get update && apt-get install -y curl zstd lshw

echo "Installing Ollama Server..."
curl -fsSL https://ollama.com/install.sh | sh

# 1. Start Ollama in the background
echo "🔴 Starting Ollama Server..."
OLLAMA_NUM_PARALLEL=24 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=1h nohup ollama serve > ollama.log 2>&1 &

# 2. Wait for Ollama to wake up
echo "⏳ Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done
echo "🟢 Ollama is running!"

# 3. Pull models if they are missing
# (We check for one common model to save time, or you can force pull all)
echo "📦 Checking/Pulling models..."
ollama pull granite4:3b
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
ollama pull qwen3:30b-instruct
ollama pull llama3.1:8b
ollama pull ministral-3:14b
ollama pull gemma3:27b

echo "Installing pip dependencies"
pip install -r requirements.txt