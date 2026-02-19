import unittest
import os
import shutil
import json
from dataclasses import asdict
from datetime import datetime
import sys

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from backend.src.database import init_db, get_db_connection
from backend.src.models.db_models import Sample, DeviceGroup, Device, LayerStructure, Measurement
from backend.src.repositories.sample_repository import SampleRepository
from backend.src.repositories.measurement_repository import MeasurementRepository

TEST_DB_PATH = "backend/data/test_expdata.db"

class TestRepositories(unittest.TestCase):
    def setUp(self):
        # Setup Test DB
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        
        # Initialize DB with test path
        # We need to hack init_db or pass path. 
        # init_db accepts db_path.
        init_db(TEST_DB_PATH)
        
        # Monkey patch repositories to use test db?
        # The repositories import get_db_connection which uses a default.
        # We should probably modify get_db_connection to accept a path or GLOBAL config,
        # but for now let's just use the fact that get_db_connection accepts db_path 
        # BUT the repositories don't pass it in their methods.
        
        # Wait, the repositories implementation:
        # def get_by_id(self, sample_id):
        #    conn = get_db_connection() 
        # It uses the default!
        
        # To test properly without changing code, we can swap the defaults in the module
        import backend.src.database
        self.original_db_path = backend.src.database.DB_PATH
        backend.src.database.DB_PATH = TEST_DB_PATH
        
        self.sample_repo = SampleRepository()
        self.measurement_repo = MeasurementRepository()

    def tearDown(self):
        # Restore default
        import backend.src.database
        backend.src.database.DB_PATH = self.original_db_path
        
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_sample_crud(self):
        # Create Data
        structures = [LayerStructure(material="Si", thick_nm_variation=None)]
        device1 = Device(device_id="d1", area_um2=100.0)
        dg = DeviceGroup(coord=(0,0), thick_nm={}, shared_properties={}, devices=[device1])
        
        sample = Sample(
            id="s1",
            name="Sample 1",
            device_type="test",
            structures=structures,
            device_groups=[dg]
        )
        
        # Insert
        self.sample_repo.insert(sample)
        
        # Get
        fetched = self.sample_repo.get_by_id("s1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "Sample 1")
        self.assertEqual(len(fetched.device_groups), 1)
        self.assertEqual(fetched.device_groups[0].devices[0].device_id, "d1")
        
        # Get Devices (Flatten)
        devices = self.sample_repo.get_devices("s1")
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].area_um2, 100.0)

    def test_measurement_crud(self):
        # Ensure sample exists for FK
        self.test_sample_crud() 
        
        meas = Measurement(
            id="m1",
            sample_id="s1",
            device_id="d1",
            measurement_type="IV",
            metadata={"temp": 300},
            data={"current": [1, 2]},
            derived={"R": 50},
            file_ref="/tmp/file"
        )
        
        # Insert
        self.measurement_repo.insert(meas)
        
        # Get by ID
        fetched = self.measurement_repo.get_by_id("m1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.measurement_type, "IV")
        self.assertEqual(fetched.metadata["temp"], 300)
        
        # Find by Device
        results = self.measurement_repo.find_by_device("s1", "d1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "m1")
        
        # Find by Sample
        results_s = self.measurement_repo.find_by_sample("s1")
        self.assertEqual(len(results_s), 1)

if __name__ == "__main__":
    unittest.main()
