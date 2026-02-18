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

from ..services.data_loader import load_log_ra_v_data
from ..services.interactive_plotter import create_log_ra_v_plot
import yaml

@router.get("/plot/{plot_id}")
def get_plot(plot_id: str):
    """
    Generic endpoint to load a plot by ID (key in YAML).
    Searches known YAML files for the key.
    """
    # Define known data files to search
    # In a real app, maybe scan the directory.
    # For now, we know the files.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    data_dir = os.path.join(project_root, "data")
    
    # Files to check
    yaml_files = ["ps_ra_data.yaml", "iv_plot_data.yaml"]
    
    found_file = None
    plot_type = None
    
    for yf in yaml_files:
        path = os.path.join(data_dir, yf)
        if not os.path.exists(path):
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
                if payload and plot_id in payload:
                    found_file = path
                    plot_data = payload[plot_id]
                    plot_type = plot_data.get("plot_type")
                    break
        except Exception:
            continue
            
    if not found_file:
         raise HTTPException(status_code=404, detail=f"Plot ID '{plot_id}' not found in data files")

    if plot_type == "ps-ra":
        # Current logic for ps-ra is slightly specific to a structure, 
        # but let's assume we can reuse the loader if adapted, or keep the old endpoint for that.
        # The user's request for "log_ra_v" is the main focus.
        # If plot_type is ps-ra, we might need to adapt `load_ps_ra_data` to take a key?
        # The existing `load_ps_ra_data` loads the WHOLE file and returns a list of series.
        # It doesn't filter by key. 
        # But `ps_ra_data.yaml` has `cofe-ps-ra` as the key.
        # Let's support `log_ra_v` first.
        pass
        
    if plot_type == "log_ra_v":
        series_list = load_log_ra_v_data(found_file, plot_id)
        if not series_list:
             raise HTTPException(status_code=404, detail="No data found for plot")
        
        # Determine title from ID or config?
        title = plot_id
        fig_json = create_log_ra_v_plot(series_list, title=title)
        
        # User requested PR text? "PR用の文章も一緒に出力"
        # We can return a wrapper object: { plot: fig_json, description: ... }
        # But existing frontend expects pure plotly json? 
        # If I change the return type, I might break frontend if it uses this endpoint.
        # But this is a NEW endpoint `/plot/{plot_id}`.
        
        response = {
            "plot": fig_json,
            "description": f"Log RA vs V plot for {plot_id}",
            "plot_type": plot_type
        }
        return response

    raise HTTPException(status_code=400, detail=f"Unknown or unsupported plot type: {plot_type}")
