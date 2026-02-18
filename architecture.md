# Project Architecture

This document outlines the folder structure, file responsibilities, and design decisions of the `expdata-plotter` application.

## Directory Structure

### Root (`expdata-plotter/`)

| File / Folder | Description |
| :--- | :--- |
| **`experiments/`** | Contains Jupyter Notebooks for interactive data analysis and plotting. These notebooks are the primary interface for researchers to explore data manually. |
| **`backend/`** | The Python backend application (FastAPI). Handles data processing and API endpoints for the web app. |
| **`frontend/`** | The React/Next.js frontend application. Visualizes data in the web browser. |
| **`data/`** | Stores raw experimental data files (YAML, CSV). Ignored by Git to keep the repository clean. |
| `start_app.sh` | Utility script to launch both the backend and frontend servers simultaneously. |
| `README.md` | General usage instructions and setup guide. |

---

### Backend (`backend/src/`)

The backend is organized into a modular structure to separate concerns between data loading, processing, and API handling.

| File / Folder | Role & Responsibility |
| :--- | :--- |
| **`main.py`** | **Entry Point**. Initializes the FastAPI application, configures CORS, and includes the API router. |
| **`api/`** | **API Layer**. Defines the HTTP endpoints exposed to the frontend. |
| &nbsp;&nbsp;`endpoints.py` | Contains route handlers (e.g., `/api/plots/ps-ra`). It calls services to get data and plot configurations, then returns JSON to the frontend. |
| **`models/`** | **Data Models**. Defines Python dataclasses/Pydantic models for type safety. |
| &nbsp;&nbsp;`analysis_types.py` | Defines core types like `RAPsData`, `RAPsPoint`, `RAPsSeries`. Used across services to ensure consistent data structures. |
| **`services/`** | **Business Logic**. Contains the core functionality for data processing and plotting. |
| &nbsp;&nbsp;`data_loader.py` | **Data Access**. Responsible for reading raw files (CSV, YAML) and parsing them into Model objects. |
| &nbsp;&nbsp;`interactive_plotter.py` | **Interactive Plotting**. Generates Plotly figure configurations (JSON) for the web frontend and interactive notebooks. |
| &nbsp;&nbsp;`cascade_iv_service.py` | **Static Plotting**. Matplotlib logic for Cascade IV curves (used in notebooks). |
| &nbsp;&nbsp;`hanle_service.py` | **Static Plotting**. Matplotlib logic for Hanle effect analysis (used in notebooks). |
| &nbsp;&nbsp;`ra_ps_service.py` | **Static Plotting**. Matplotlib logic for RA-Ps summary plots (used in notebooks). |

---

### Frontend (`frontend/`)

The frontend is a Next.js application used for viewing interactive plots.

| File / Folder | Role & Responsibility |
| :--- | :--- |
| **`src/app/`** | **Pages & Routing**. Defines the application routes and page layouts. |
| **`src/components/`** | **UI Components**. Reusable React components (e.g., Plot viewers, Layout wrappers). |
| `package.json` | API dependencies and scripts (`npm run dev`). |

---

### Design Decisions

1.  **Shared Logic**: The logic for loading data (`data_loader.py`) and defining data structures (`models/`) is shared between the API (Web App) and the Notebooks (`experiments/`). This ensures consistent results regardless of the interface used.
2.  **Stateless API**: The backend endpoints are stateless. They read data from the disk on request, allowing researchers to update data files and immediately see results without restarting the server.
3.  **Visualization Separation**:
    - **Matplotlib** (`*_service.py`) is used for static, publication-quality figures in notebooks.
    - **Plotly** (`interactive_plotter.py`) is used for interactive data exploration in both the Web App and Notebooks.
