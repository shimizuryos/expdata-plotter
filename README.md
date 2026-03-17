# Experimental Data Plotter

This repository contains tools for analyzing and plotting experimental data, including Cascade IV curves, Hanle effect measurements, and RA-Ps summaries.

## Directory Structure

- **`experiments/`**: Jupyter notebooks for interactive analysis and plotting.
  - `plot_cascade_iv.ipynb`: Analyze IV characteristics.
  - `plot_hanle.ipynb`: Analyze Hanle effect data.
  - `plot_ps_ra.ipynb`: Visualizes RA vs Ps summary data.
- **`data/`**: Directory for storing raw data files (YAML, CSV, etc.).
  - Note: This directory is git-ignored. Place your local data files here.
- **`backend/src/`**: Core logic and utility functions.
  - `models/`: strict Pydantic SI data structures (e.g., `RAPsSeries`, `HanleBroadSeries`) and Database hierarchy (Sample > Device > Measurement).
  - `services/`: Data loading, physics fitting (`ps_ra_fitting.py`), and plotting logic.
  - `utils/`: Centralized single-dimension unit mappings (`units.py`).
- **`frontend/`**: Web application frontend (Next.js).

## Setup

1.  **Install Dependencies**:
    Ensure you have Python installed, then install the required packages:
    ```bash
    pip install -r backend/requirements.txt
    ```
    *(Note: If `backend/requirements.txt` does not exist, install: `pandas`, `matplotlib`, `plotly`, `pyyaml`, `scipy`, `numpy`, `jupyter`)*

2.  **Environment**:
    It is recommended to use a virtual environment or Conda environment.

## Usage

### 1. Web Application (Interactive Plotter)

**Quick Start (Recommended)**:
The `start_app.sh` script will automatically:
1.  Check for Conda.
2.  Create/Activate a `data-plotter` environment.
3.  Install all Python and Node.js dependencies.
4.  Launch the app.

```bash
./start_app.sh
```
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:8000](http://localhost:8000)

**Navigation**:
The application Home Page provides access to:
- **PS-RA Plot**: Visualizes point data (RA, Ps, RMS) from `data/ps_ra_data.yaml`.
- **Log RA vs V Plot**: Visualizes Log RA vs Voltage traces. configured in `data/iv_plot_data.yaml`.
    - Supports grouping of IV curves (e.g. `low_ps`, `high_ps`) with custom colors.
    - Calculates RA product (Resistance * Area) automatically.
- **Cascade IV**: Visualizes IV curves (requires data).
- **Hanle**: Visualizes Hanle effect data (requires data).

### Configuration (Log RA vs V)

To add a new Log RA vs V plot, create or edit `data/iv_plot_data.yaml`:

```yaml
my_plot_key:
  plot_type: log_ra_v
  group_name_1:
    color: "red"
    data:
      sample_label_1:
        file_path: '/absolute/path/to/iv_data.txt'
        area: 250
      sample_label_2:
        file_path: '/absolute/path/to/iv_data_2.txt'
        area: 250
  group_name_2:
    color: "blue"
    data:
       ...
```

- **Upload**: Utility to upload files to `backend/data/raw` (if needed).

**Troubleshooting**:
- **404 Not Found (Plots)**: If you see a 404 error when opening a plot page, it usually means the backend cannot find the required data file.
    - Check that `data/ps_ra_data.yaml` exists in the `expdata-plotter` root.
    - Check the backend logs in your terminal for "Data file not found" messages.

**Manual Start**:
If you prefer running them separately:

**Frontend (Client)**:
1.  Navigate to `frontend/`: `cd frontend`
2.  Install dependencies: `npm install`
3.  Run the development server: `npm run dev`

**Backend (Server)**:
1.  Navigate to `backend/`: `cd backend`
2.  Run the API server: `python src/main.py`
    - Ensure your data is in `../data/ps_ra_data.yaml` relative to backend.

### 2. Notebook Analysis
- **`experiments/plot_ps_ra.ipynb`**: Detailed analysis and static/interactive plotting in Jupyter.
- **`experiments/plot_cascade_iv.ipynb`**: IV curve analysis.

## Development

## Development

- **Data Models**: Pydantic validated SI unit models in `backend/src/models/analysis_types.py` and Database mappings in `db_models.py`.
- **Plotting Logic**: 
  - `backend/src/services/interactive_plotter.py` (Plotly)
  - `backend/src/services/ra_ps_service.py` (Matplotlib)
  - `backend/src/services/hanle_service.py` (Matplotlib)
  - `backend/src/services/cascade_iv_service.py` (Matplotlib)
- **Data Loading**: All loading logic is centralized in `backend/src/services/data_loader.py`, routed to explicit types securely natively in SI magnitudes.
- **Unit Management**: Unified through pure mathematical conversions in `backend/src/utils/units.py`.

## Git Workflow
- The `data/` and `outputs/` directories are ignored by git to prevent committing large raw files or sensitive data.
- Commit your notebooks (`experiments/`) and code changes (`backend/`).
