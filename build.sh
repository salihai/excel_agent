set -e

OLLAMA_DIR="$HOME/ollama"
mkdir -p "$OLLAMA_DIR"

curl -fsSL https://ollama.com/download/Ollama-linux.tar.gz -o "$OLLAMA_DIR/Ollama-linux.tar.gz"

tar -xzf "$OLLAMA_DIR/Ollama-linux.tar.gz" -C "$OLLAMA_DIR"
rm "$OLLAMA_DIR/Ollama-linux.tar.gz"

export PATH="$OLLAMA_DIR:$PATH"

ollama pull llama2

nohup ollama serve > "$OLLAMA_DIR/ollama.log" 2>&1 &