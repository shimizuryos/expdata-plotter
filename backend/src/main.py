from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router
import uvicorn

app = FastAPI(title="Research Data App API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    # Running from backend/ directory as 'python -m src.main'
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
