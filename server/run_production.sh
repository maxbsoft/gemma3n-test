#!/bin/bash

# Gemma 3n API Server - Production Mode
# Запуск сервера в фоновом режиме с логированием

set -e

echo "🚀 Starting Gemma 3n API Server - Production Mode"
echo "=================================================="

# Configuration
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
MODEL=${MODEL:-"gemma-3n-e2b-quantized"}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-"info"}

# Directories
LOG_DIR="logs"
PID_FILE="$LOG_DIR/server.pid"
ACCESS_LOG="$LOG_DIR/access.log"
ERROR_LOG="$LOG_DIR/error.log"
SERVER_LOG="$LOG_DIR/server.log"

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
        --stop)
            echo "Stopping server..."
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
                echo "⚠️  PID file not found, server may not be running"
            fi
            exit 0
            ;;
        --status)
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
        --logs)
            echo "📋 Showing recent logs:"
            echo "=================="
            echo "📊 Server Log (last 20 lines):"
            tail -n 20 "$SERVER_LOG" 2>/dev/null || echo "No server logs found"
            echo ""
            echo "🔴 Error Log (last 10 lines):"
            tail -n 10 "$ERROR_LOG" 2>/dev/null || echo "No error logs found"
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
            echo "  --stop              Stop the server"
            echo "  --status            Check server status"
            echo "  --logs              Show recent logs"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Start server with defaults"
            echo "  $0 --model gemma-3n-e4b-full --workers 2"
            echo "  $0 --stop           # Stop server"
            echo "  $0 --status         # Check status"
            echo "  $0 --logs           # View logs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "❌ Server is already running (PID: $PID)"
        echo "Use '$0 --stop' to stop it first"
        exit 1
    else
        echo "⚠️  Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: ../.env file not found"
    echo "   Please create .env file with your HF_TOKEN"
fi

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  Log Directory: $LOG_DIR"
echo "  PID File: $PID_FILE"

echo ""
echo "Starting server in background..."

# Start server with nohup and logging
nohup python start_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --no-check \
    > "$SERVER_LOG" 2> "$ERROR_LOG" &

# Save PID
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

echo "✅ Server started in background"
echo "   PID: $SERVER_PID"
echo "   Logs: $SERVER_LOG"
echo "   Errors: $ERROR_LOG"
echo ""
echo "🌐 Server will be available at:"
echo "   API: http://$HOST:$PORT"
echo "   Docs: http://$HOST:$PORT/docs"
echo "   Health: http://$HOST:$PORT/v1/health"
echo ""
echo "📋 Commands:"
echo "   Check status: $0 --status"
echo "   View logs: $0 --logs"
echo "   Stop server: $0 --stop"

# Wait a bit and check if server started successfully
sleep 5
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "🎉 Server started successfully!"
    
    # Try to get health status
    sleep 10
    echo "📊 Health check:"
    curl -s http://localhost:$PORT/v1/health | python3 -m json.tool 2>/dev/null || echo "⏳ Server still starting up..."
else
    echo ""
    echo "❌ Server failed to start. Check error logs:"
    cat "$ERROR_LOG"
    rm -f "$PID_FILE"
    exit 1
fi 