from fastapi import APIRouter, HTTPException
from ..services.data_loader import load_ps_ra_data
from ..services.interactive_plotter import create_ps_ra_plot, create_iv_plot, create_hanle_plot
import os
import shutil
from fastapi import UploadFile, File

router = APIRouter()

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@router.get("/plots/ps-ra")
def get_ps_ra_plot():
    # Resolve path relative to this file to avoid CWD dependency
    # File: backend/src/api/endpoints.py
    # Goal: expdata-plotter/data/ps_ra_data.yaml
    # Path: ../../../data/ps_ra_data.yaml (relative to file)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    data_path = os.path.join(project_root, "data/ps_ra_data.yaml")
    
    if not os.path.exists(data_path):
         # Try fallback for local dev if structure differs
         # e.g. if running from root
         fallback = os.path.join(os.getcwd(), "data/ps_ra_data.yaml")
         if os.path.exists(fallback):
             data_path = fallback
             
    if not os.path.exists(data_path):
         raise HTTPException(status_code=404, detail=f"Data file not found at {data_path}")

    # load_ps_ra_data now returns List[RAPsSeries]
    # create_ps_ra_plot accepts list[RAPsSeries]
    series_list = load_ps_ra_data(data_path)
    if not series_list:
        raise HTTPException(status_code=404, detail="No data found in yaml")
        
    fig_json = create_ps_ra_plot(series_list)
    return fig_json

@router.get("/plots/iv")
def get_iv_plot():
    # Placeholder: In future, load data/exp_data.csv or similar
    fig_json = create_iv_plot(None)
    return fig_json

@router.get("/plots/hanle")
def get_hanle_plot():
    # Placeholder
    fig_json = create_hanle_plot(None)
    return fig_json
