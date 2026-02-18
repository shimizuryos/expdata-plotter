#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID
        echo "Backend stopped."
    fi
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "========================================"
echo "Starting Research Data App"
echo "========================================"

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found."
    echo "Please install Anaconda or Miniconda."
    exit 1
fi

ENV_NAME="data-plotter"

# Initialize Conda (required for script usage)
# Try to find conda.sh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' exists."
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.11 -y
fi

# Activate Environment
echo "Activating '$ENV_NAME'..."
conda activate $ENV_NAME

# Install Backend Dependencies
if [ -f "backend/requirements.txt" ]; then
    echo "Installing/Updating backend dependencies..."
    pip install -r backend/requirements.txt
fi

# Install Frontend Dependencies
echo "Installing/Updating frontend dependencies..."
cd frontend
npm install
cd ..

# Start Backend
echo "[1/2] Starting Backend..."
cd backend
export PYTHONPATH=$PYTHONPATH:.
python -m src.main &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to initialize
sleep 3

# Start Frontend
echo "[2/2] Starting Frontend..."
cd frontend
npm run dev

# Wait for background processes
wait
