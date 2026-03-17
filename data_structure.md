# Data Structure & Hierarchy

This document explains the data architecture inside `expdata-plotter`, specifically focusing on the hierarchical relationships, coordinate mapping, and strict internal unit protocols.

## Hierarchical Database Model

The backend stores experiment metadata and files using a 4-tier hierarchy:

1. **Sample**: The top-level entity representing a physical wafer or piece.
    - Defines the coordinate bounds of the physical layout (default `max_x=22`, `max_y=22`).
    - Holds macroscopic shared properties like `Structures` and `r_parasitic`.
2. **DeviceGroup**: A logical batch of devices that share specific layer thicknesses (`thick_nm`) or fabrication properties.
3. **Device**: The physical microscopic junction/element where measurements happen.
    - **Spatial Coordinates**: Every device is explicitly assigned an `(x_coord, y_coord)` tuple. This maps its location directly to the 22x22 physical grid on the Sample.
    - Specifies the individual physical area.
4. **Measurement**: Inherits the `device_id` and records the exact experimental run.
    - Raw data relies on a `file_ref`.
    - Handles derived values structurally to bypass expensive parses on the fly.

## SI Internal Unit Standard

To avoid "magnitude errors" bridging UI scaling with scientific models (e.g., $nm$ vs $cm$, $\Omega \cdot m^2$ vs $\Omega \cdot \mu m^2$), the backend adheres to a strict protocol:

- **Inside the Database**: All numerical columns and `Measurement.derived` blobs natively represent properties in exact **Standard International (SI) units** ($V$, $A$, $m^2$, $\Omega \cdot m^2$, $T$).
- **Inside Plotting/Fitting Models**: Pydantic validators (`analysis_types.py`) enforce unit bounds and scaling. `LikelihoodCalculator` physics equations receive pure SI tensors.
- **Frontend / Display**: Human-readable formats ($\mu m^2$, $\%$) are isolated strictly to UI components and format responses built by the `interactive_plotter.py`.

## Hanle Disambiguation Models

Hanle files often exhibit three flavors that conflict structurally:
- **HanleRawSeries**: The unparsed 2-column $(T, V)$ dump.
- **HanleBroadSeries**: Broad scans requiring $A\_b3t$, $W\_b3t$, and side-lobe models.
- **HanleNarrowSeries**: 3-terminal narrow sweeps emphasizing the center peak.

These are explicitly separated into independent Pydantic classes and endpoints, preventing heterogeneous assumptions during fitting rounds.
