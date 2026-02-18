import sys
import os
import json

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from backend.src.repositories.sample_repository import SampleRepository
from backend.src.repositories.measurement_repository import MeasurementRepository

def verify_db():
    print("--- Verifying DB Content ---")
    
    sample_repo = SampleRepository()
    measurement_repo = MeasurementRepository()
    
    # Check Sample
    sample_id = "250918_cofe_3t"
    sample = sample_repo.get_by_id(sample_id)
    
    if sample:
        print(f"✅ Found Sample: {sample.name} ({sample.id})")
        print(f"   Device Groups: {len(sample.device_groups)}")
        devices = sample_repo.get_devices(sample_id)
        print(f"   Total Devices: {len(devices)}")
        if devices:
            print(f"   Sample Device: {devices[0]}")
    else:
        print(f"❌ Sample {sample_id} NOT found!")
        
    # Check Measurements
    measurements = measurement_repo.find_by_sample(sample_id)
    print(f"✅ Found {len(measurements)} measurements for sample {sample_id}")
    
    hanle_meas = [m for m in measurements if m.measurement_type == "Hanle"]
    iv_meas = [m for m in measurements if m.measurement_type == "IV"]
    
    print(f"   Hanle (derived): {len(hanle_meas)}")
    print(f"   IV (raw refs): {len(iv_meas)}")
    
    if hanle_meas:
        print(f"   Sample Hanle: ID={hanle_meas[0].id}, Derived={hanle_meas[0].derived}")

    if iv_meas:
        print(f"   Sample IV: ID={iv_meas[0].id}, FileRef={iv_meas[0].file_ref}")

if __name__ == "__main__":
    verify_db()
