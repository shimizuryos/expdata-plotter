from typing import List, Tuple, Sequence, Optional, Dict
import pandas as pd
import re
import yaml
from ..models.analysis_types import ParsedIVSeries, RAPsSeries, RAPsPoint

def load_iv_data(file_path: str) -> ParsedIVSeries:
    """
    Load IV data file and return ParsedIVSeries.
    Supports simple space-separated format (skip first 2 lines).
    """
    id_milliamp: list[float] = []
    vd_millivolt: list[float] = []
    resistance_ohm: list[float] = []
    warnings: list[str] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_index, raw_line in enumerate(f, start=1):
                if line_index <= 2:
                    continue
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    warnings.append(f"line {line_index}: Not enough columns")
                    continue

                try:
                    id_amp = float(parts[0])
                    vd_volt = float(parts[1])
                    r_ohm = float(parts[4])
                except ValueError:
                    warnings.append(f"line {line_index}: Cannot parse numbers")
                    continue

                id_milliamp.append(id_amp * 1_000.0)
                vd_millivolt.append(vd_volt * 1_000.0)
                resistance_ohm.append(r_ohm)
    except FileNotFoundError:
        warnings.append(f"File not found: {file_path}")
    except Exception as e:
        warnings.append(f"Error reading file: {e}")

    return ParsedIVSeries(
        id_mA=id_milliamp,
        vd_mV=vd_millivolt,
        r_ohm=resistance_ohm,
        warnings=warnings,
    )

def parse_iv_csv(file_path: str) -> ParsedIVSeries:
    """
    Load IV CSV data with 'DataName' / 'DataValue' structure.
    """
    warnings: List[str] = []
    id_list: List[float] = []
    vd_list: List[float] = []
    r_list: List[float] = []

    try:
        df = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            dtype=str,
            engine="python",
            names=list(range(256)),
            skip_blank_lines=True,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(f"CSV load failed: {exc}")

    # DataName check
    name_rows = df.index[df[0].astype(str).str.strip() == "DataName"].tolist()
    if not name_rows:
        raise ValueError("DataName row not found")
    name_idx = name_rows[0]
    names = df.loc[name_idx, 1:].tolist()
    names = [("" if pd.isna(x) else str(x).strip()) for x in names]

    try:
        name_to_idx = {name: i for i, name in enumerate(names)}
        id_idx = name_to_idx["Id"]
        vd_idx = name_to_idx["Vd"]
        r_idx = name_to_idx["R"]
    except KeyError as exc:
        raise ValueError(f"Missing required columns: {exc}")

    header_len = len(names)

    def to_float(token: str) -> Optional[float]:
        if token is None:
            return None
        t = str(token).strip()
        if t == "" or t.lower() == "nan":
            return None
        try:
            return float(t)
        except ValueError:
            return None

    value_df = df[df[0].astype(str).str.strip() == "DataValue"]
    for i, row in value_df.iterrows():
        values = row[1:].tolist()
        if len(values) < header_len:
            values += [""] * (header_len - len(values))

        id_val = to_float(values[id_idx])
        vd_val = to_float(values[vd_idx])
        r_val = to_float(values[r_idx])

        if id_val is None or vd_val is None or r_val is None:
            # warnings.append(f"line {i+1}: Numeric conversion failed")
            continue

        id_list.append(id_val * 1e3)
        vd_list.append(vd_val * 1e3)
        r_list.append(r_val)

    if not id_list:
        raise ValueError("No valid DataValue rows found")

    return ParsedIVSeries(id_mA=id_list, vd_mV=vd_list, r_ohm=r_list, warnings=warnings)

def read_hanle_raw_data(file_path: str) -> Tuple[List[float], List[float]]:
    """
    Simple Hanle reader (skip 1st line, take first 2 cols).
    Returns (magnetic_field_Oe, voltage_uV).
    """
    magnetic_field_Oe: List[float] = []
    voltage_uV: List[float] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            cols = text.split()
            if len(cols) < 2:
                continue
            try:
                magnetic_field_Oe.append(float(cols[0]))
                voltage_uV.append(float(cols[1])*1_000_000)
            except ValueError:
                continue

    return magnetic_field_Oe, voltage_uV

def read_hanle_data(file_path: str) -> List[List[List[float]]]:
    """
    Multi-section Hanle reader (sections separated by '==').
    Returns list of [magnetic_field_list, voltage_list].
    """
    def is_delim(line_text: str) -> bool:
        s = line_text.strip()
        if not s:
            return False
        return s.replace(" ", "") == "=="

    series_list: List[List[List[float]]] = []
    current_b: List[float] = []
    current_v: List[float] = []

    def flush_current_if_needed():
        if current_b and current_v and len(current_b) == len(current_v):
            series_list.append([current_b.copy(), current_v.copy()])

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()

                if is_delim(text):
                    flush_current_if_needed()
                    current_b.clear()
                    current_v.clear()
                    continue

                if not text or text.startswith("#"):
                    continue

                cols = text.split()
                if len(cols) < 2:
                    continue
                try:
                    b_val = float(cols[0])
                    v_val = float(cols[1])
                except ValueError:
                    continue

                current_b.append(b_val)
                current_v.append(v_val * 1_000_000)

        flush_current_if_needed()
    except FileNotFoundError:
        pass

    return series_list

def read_hanle_broad(
    file_path: str,
) -> Tuple[Dict[str, float], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Extracts params and specific data sections for Hanle Broad.
    Returns (params_dict, exp_data, fitting_data, broad_fitting_data).
    """
    # Parse header for parameters
    params = {}
    
    def _parse_header_params(path: str) -> Dict[str, float]:
        p = {}
        def _find_value(pattern: str, text: str) -> float:
            m = re.search(pattern, text)
            if not m:
                return float('nan')
            try:
                return float(m.group(1))
            except ValueError:
                return float('nan')

        float_pattern = r"([+\-]?\d*(?:\.\d+)?(?:[Ee][+\-]?\d+)?)"
        try:
            with open(path, "r", encoding="utf-8") as rf:
                first_line = rf.readline().strip()
            if first_line.startswith("#"):
                first_line = first_line[1:]

            p["A_b3t"]   = _find_value(rf"A_b3t\s*=\s*{float_pattern}", first_line)
            p["W_b3t"]   = _find_value(rf"W_b3t\s*=\s*{float_pattern}", first_line)
            p["A_n3t"]   = _find_value(rf"A_n3t\s*=\s*{float_pattern}", first_line)
            p["W_n3t"]   = _find_value(rf"W_n3t\s*=\s*{float_pattern}", first_line)
            p["Ts"]      = _find_value(rf"Ts\s*=\s*{float_pattern}", first_line)
            p["Voff_b"]  = _find_value(rf"Voff_b\s*=\s*{float_pattern}", first_line)
            p["Voff_n"]  = _find_value(rf"Voff_n\s*=\s*{float_pattern}", first_line)
            p["yokozure"] = _find_value(rf"yokozure\s+{float_pattern}", first_line)
            p["alpha"]    = _find_value(rf"alpha\s+{float_pattern}", first_line)
        except Exception:
            pass
        return p

    params = _parse_header_params(file_path)
    series = read_hanle_data(file_path)

    # Note: exp and fitting order is reversed in data file usually (0: fitting, 1: exp)
    if len(series) >= 2:
        exp_data = series[1]
        fitting_data = series[0]
        broad_fitting_data = []
        if len(series) >= 3:
             # Logic from original code: if 4 sections, index 3 is broad? if 3, index 2 is broad?
             # Original: 
             # if len=4: broad=series[3]
             # elif len=3: broad=series[2]
            if len(series) == 4:
                broad_fitting_data = series[3]
            else:
                broad_fitting_data = series[2]
        
        return params, exp_data, fitting_data, broad_fitting_data
    
    return params, [], [], []

def read_hanle_n_only(
    file_path: str,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Extracts (exp_data, fitting_data) from Hanle data.
    """
    series = read_hanle_data(file_path)
    if len(series) < 2:
        # raise ValueError or return empty
        return [], []
    exp_data = series[1]
    fitting_data = series[0]
    return exp_data, fitting_data

    return exp_data, fitting_data

from ..models.analysis_types import ParsedIVSeries, RAPsSeries, RAPsPoint, LogRAVSeries

def load_log_ra_v_data(yaml_path: str, plot_key: str) -> List[LogRAVSeries]:
    """
    Load Log-RA-V data from YAML for a specific plot key.
    Calculates RA = R * Area for each series.
    Returns list of LogRAVSeries.
    """
    series_list: List[LogRAVSeries] = []
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
            
        plot_config = payload.get(plot_key, {})
        if not plot_config:
            return []
            
        # Iterate over groups (keys that are not 'plot_type' etc.)
        for group_key, group_data in plot_config.items():
            if group_key in ["plot_type"]:
                continue
                
            if not isinstance(group_data, dict):
                continue
                
            color = group_data.get("color", "black")
            group_items = group_data.get("data", {})
            
            for item_key, item_data in group_items.items():
                file_path = item_data.get("file_path")
                area_um2 = item_data.get("area", 1.0)
                
                if not file_path:
                    continue
                    
                # Load IV data
                # We reuse load_iv_data or parse_iv_csv depending on file extension or format
                # The example path provided doesn't have an extension, implying simple format maybe?
                # Let's try load_iv_data first (simple space separated)
                iv_series = load_iv_data(file_path)
                
                # If load_iv_data failed (empty lists), try csv? 
                # Or maybe check warnings. load_iv_data returns warnings.
                if not iv_series.r_ohm:
                     # Fallback or error handling? 
                     # For now, if no data, skip
                     continue

                # Calculate RA
                # ra_ohm_um2 = r_ohm * area_um2
                ra_list = [r * area_um2 for r in iv_series.r_ohm]
                
                series_list.append(LogRAVSeries(
                    vd_mV=iv_series.vd_mV,
                    ra_ohm_um2=ra_list,
                    label=item_key,
                    color=color,
                    group_label=group_key
                ))

    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error loading yaml or processing data: {e}")
        
    return series_list


# ... (omitted)

def load_ps_ra_data(yaml_path: str) -> List[RAPsSeries]:
    """
    Load Ps-RA data from YAML.
    Returns list of RAPsSeries.
    """
    series_list: List[RAPsSeries] = []
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
            
        root = payload.get("data", {})
        for group_name, group in root.items():
            for key, item in group.items():
                data = item.get("data", [])
                label = item.get("label", key)
                color = item.get("color", "tab:blue")
                
                if data:
                    points = []
                    for row in data:
                        if len(row) >= 3:
                            ra_val, ps_val, rms_val = row[0], row[1], row[2]
                            p_label = row[3] if len(row) >= 4 else None
                            points.append(RAPsPoint(ra=ra_val, ps=ps_val, rms=rms_val, label=p_label))
                    
                    series_list.append(RAPsSeries(points=points, label=label, color=color))
                    
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error loading yaml: {e}")
        
    return series_list
