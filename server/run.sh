#!/bin/bash

# Gemma 3n API Server Startup Script

set -e

echo "🚀 Starting Gemma 3n API Server"
echo "================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run from server directory."
    exit 1
fi

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: ../.env file not found"
    echo "   Please create .env file with your HF_TOKEN"
    echo "   Example: echo 'HF_TOKEN=your_token_here' > ../.env"
fi

# Default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
MODEL=${MODEL:-"gemma-3n-e2b-quantized"}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-"info"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --dev|--reload)
            RELOAD="--reload"
            WORKERS=1
            shift
            ;;
        --debug)
            LOG_LEVEL="debug"
            shift
            ;;
        --stop)
            echo "Stopping server..."
            PID_FILE="logs/server.pid"
            if [ -f "$PID_FILE" ]; then
                PID=$(cat "$PID_FILE")
                if kill -0 "$PID" 2>/dev/null; then
                    echo "Killing process $PID"
                    kill -TERM "$PID"
                    rm -f "$PID_FILE"
                    echo "✅ Server stopped"
                else
                    echo "⚠️  Process $PID not found, removing stale PID file"
                    rm -f "$PID_FILE"
                fi
            else
                echo "⚠️  PID file not found. Checking for running processes..."
                pkill -f "start_server.py" && echo "✅ Server stopped" || echo "❌ No server process found"
            fi
            exit 0
            ;;
        --status)
            PID_FILE="logs/server.pid"
            if [ -f "$PID_FILE" ]; then
                PID=$(cat "$PID_FILE")
                if kill -0 "$PID" 2>/dev/null; then
                    echo "✅ Server is running (PID: $PID)"
                    echo "📊 Server status:"
                    curl -s http://localhost:$PORT/v1/health | python3 -m json.tool 2>/dev/null || echo "❌ Server not responding"
                else
                    echo "❌ Server not running (stale PID file)"
                    rm -f "$PID_FILE"
                fi
            else
                echo "❌ Server not running"
            fi
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST         Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT         Port to bind to (default: 8000)"
            echo "  --model MODEL       Model to load (default: gemma-3n-e2b-quantized)"
            echo "  --workers N         Number of workers (default: 1)"
            echo "  --dev, --reload     Enable development mode with auto-reload"
            echo "  --debug             Enable debug logging"
            echo "  --stop              Stop the server"
            echo "  --status            Check server status"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Available models:"
            echo "  - gemma-3n-e2b-quantized (fast, 4-6GB VRAM)"
            echo "  - gemma-3n-e2b-full (high quality, 8-12GB VRAM)"
            echo "  - gemma-3n-e4b-full (high quality, 8-12GB VRAM)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Start with defaults"
            echo "  $0 --model gemma-3n-e4b-full         # Use E4B model"
            echo "  $0 --dev                              # Development mode"
            echo "  $0 --port 8080 --debug               # Custom port with debug"
            echo "  $0 --stop                             # Stop server"
            echo "  $0 --status                           # Check status"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"
if [ -n "$RELOAD" ]; then
    echo "  Mode: Development (auto-reload enabled)"
else
    echo "  Mode: Production"
fi

echo ""
echo "Starting server..."
echo "API will be available at: http://$HOST:$PORT"
echo "Documentation at: http://$HOST:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
if [ -n "$RELOAD" ]; then
    # Development mode
    python start_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --log-level "$LOG_LEVEL" \
        --reload
else
    # Production mode
    python start_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi 