import yaml
import os
import sys
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from backend.src.database import init_db
from backend.src.models.db_models import Sample, DeviceGroup, Device, LayerStructure, Measurement
from backend.src.repositories.sample_repository import SampleRepository
from backend.src.repositories.measurement_repository import MeasurementRepository

def seed_db():
    print("Initializing Database...")
    init_db()
    
    sample_repo = SampleRepository()
    measurement_repo = MeasurementRepository()

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    ps_ra_path = os.path.join(project_root, "data/ps_ra_data.yaml")
    iv_path = os.path.join(project_root, "data/iv_plot_data.yaml")

    print(f"Loading data from {ps_ra_path} and {iv_path}...")
    
    # 1. Create Sample "250918_cofe_3t" (Hardcoded based on user request context/existing data)
    # The existing YAMLs don't fully define the derived structure, so we infer/hardcode the sample skeleton
    # and populate devices from the data files.
    
    sample_id = "250918_cofe_3t"
    
    structures = [
        LayerStructure(material="Si", thick_nm_variation=None),
        LayerStructure(material="MgO", thick_nm_variation=[1.0, 2.0]), # Inferred
        LayerStructure(material="CoFe", thick_nm_variation=None)
    ]
    
    # We will build device_groups dynamically by parsing the YAMLs to find all unique devices
    # and their coordinates/areas.
    # Currently YAMLs have devices like "7-10-c", "7-10-d-left"
    # Format seems to be "{x}-{y}-{type}" or "{x}-{y}-{type}-{suffix}"
    
    devices_dict = {} # (x, y) -> list of devices
    
    # Helper to parse device ID
    def parse_device_id(d_id: str):
        parts = d_id.split("-")
        try:
            x = int(parts[0])
            y = int(parts[1])
            return x, y
        except ValueError:
            print(f"Warning: Could not parse coordinates from device_id {d_id}")
            return 0, 0

    # LOAD PS-RA DATA
    if os.path.exists(ps_ra_path):
        with open(ps_ra_path, "r") as f:
            ps_ra_data = yaml.safe_load(f)
            
        # Structure: data -> 250918_cofe_3t -> categories -> data -> list of lists
        # list: [ra, ps, rms, label]
        
        sample_data = ps_ra_data.get("data", {}).get(sample_id, {})
        
        for category, content in sample_data.items():
            if not isinstance(content, dict): continue
            
            raw_list = content.get("data", [])
            for row in raw_list:
                # row: [ra, ps, rms, label]
                # label often contains " 4 K" suffix
                ra, ps, rms, label = row[0], row[1], row[2], row[3]
                
                clean_device_id = label.replace(" 4 K", "").replace(" 4K", "").strip()
                x, y = parse_device_id(clean_device_id)
                
                # Register device
                if (x, y) not in devices_dict:
                    devices_dict[(x, y)] = []
                
                # Check if device already in list (avoid dups)
                if not any(d.device_id == clean_device_id for d in devices_dict[(x, y)]):
                    # Area is not in PS-RA yaml, defaults to 0 (will update from IV if avail)
                    devices_dict[(x, y)].append(Device(device_id=clean_device_id, area_um2=0.0))

                # Create/Insert Measurement (Derived Hanle/Summary)
                meas_id = f"meas_{clean_device_id}_hanle_summary_{int(datetime.now().timestamp())}"
                
                # This seems to be a SUMMARY measurement (Ps/RA), likely from Hanle or similar.
                # User's schema has measurement_type="Hanle" storing "derived" fields.
                
                measurement = Measurement(
                    id=meas_id,
                    sample_id=sample_id,
                    device_id=clean_device_id,
                    measurement_type="Hanle", # Assuming source
                    metadata={"temp_K": 4.0}, # inferred from label
                    data=None, # summary only
                    derived={
                        "ra_ohm_um2": ra,
                        "ps_percent": ps,
                        "rms": rms
                    },
                    file_ref=None # source file unknown in this yaml
                )
                try:
                    measurement_repo.insert(measurement)
                    print(f"Inserted Hanle Summary for {clean_device_id}")
                except Exception as e:
                    print(f"Error inserting measurement {meas_id}: {e}")

    # LOAD IV DATA
    if os.path.exists(iv_path):
        with open(iv_path, "r") as f:
            iv_data = yaml.safe_load(f)
            
        # Structure: generic keys -> plot_type -> groups -> data -> device_id -> {file_path, area}
        
        for plot_key, plot_config in iv_data.items():
            if plot_config.get("plot_type") != "log_ra_v": continue
            
            for group_key, group_val in plot_config.items():
                if not isinstance(group_val, dict) or "data" not in group_val: continue
                
                for dev_id, info in group_val["data"].items():
                    # clean dev_id? usually keys in IV yaml are clean like "7-10-c"
                    clean_device_id = dev_id
                    area = info.get("area", 0.0)
                    file_path = info.get("file_path")
                    
                    x, y = parse_device_id(clean_device_id)
                    
                    # Update or Add device
                    if (x, y) not in devices_dict:
                        devices_dict[(x, y)] = []
                    
                    existing_dev = next((d for d in devices_dict[(x, y)] if d.device_id == clean_device_id), None)
                    if existing_dev:
                        existing_dev.area_um2 = area # Update area from IV data
                    else:
                        devices_dict[(x, y)].append(Device(device_id=clean_device_id, area_um2=area))
                        
                    # Create Measurement
                    meas_id = f"meas_{clean_device_id}_iv_4k_{int(datetime.now().timestamp())}"
                    
                    measurement = Measurement(
                        id=meas_id,
                        sample_id=sample_id,
                        device_id=clean_device_id,
                        measurement_type="IV",
                        metadata={"temp_K": 4.0}, # Inferred from context (often 4K plot)
                        data=None, # Raw data would be loaded from file_path, skipping for seed
                        derived=None,
                        file_ref=file_path
                    )
                     # Only insert if ID distinct or handle collision. 
                     # current ID logic is weak (timestamp based), but okay for one-shot seed.
                     # Better to hash file path or something.
                    
                    # Ensure unique ID for this execution
                    import hashlib
                    hash_input = f"{clean_device_id}_IV_{file_path}".encode()
                    meas_id = hashlib.md5(hash_input).hexdigest()
                    measurement.id = meas_id
                    
                    try:
                        measurement_repo.insert(measurement)
                        print(f"Inserted IV Measurement for {clean_device_id}")
                    except Exception as e:
                        print(f"Error inserting measurement {meas_id}: {e}")

    # Construct Sample Object via DeviceGroups
    device_groups = []
    for coord, devs in devices_dict.items():
        dg = DeviceGroup(
            coord=coord,
            thick_nm={"MgO": 1.0}, # Dummy/Default
            shared_properties={"parasitic_resistance_ohm": 0.0}, # Default
            devices=devs
        )
        device_groups.append(dg)
        
    sample = Sample(
        id=sample_id,
        name="250918 CoFe 3T",
        device_type="three_terminal_hanle",
        structures=structures,
        device_groups=device_groups,
        note="Imported from YAMLs"
    )
    
    sample_repo.insert(sample)
    print(f"Inserted Sample {sample.id} with {len(device_groups)} device groups.")
    
    print("Seeding Complete.")

if __name__ == "__main__":
    seed_db()
