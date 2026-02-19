from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router
from .api.data_router import router as data_router
from .api.utils_router import router as utils_router
from .database import init_db
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

# Initialize DB on startup
init_db()

app.include_router(router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(utils_router, prefix="/api")

if __name__ == "__main__":
    # Running from backend/ directory as 'python -m src.main'
    uvicorn.run(app, host="0.0.0.0", port=8000)
