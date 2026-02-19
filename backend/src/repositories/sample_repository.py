import json
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from ..database import get_db_connection, get_db_cursor
from ..models.db_models import Sample, DeviceGroup, Device, LayerStructure

class SampleRepository:
    def insert(self, sample: Sample) -> None:
        """
        Inserts a new sample into the database.
        """
        self._save_sample(sample)

    def _save_sample(self, sample: Sample) -> None:
        """
        Internal method to save sample to DB.
        """
        # Convert nested dataclasses to dicts for JSON serialization
        structures_json = json.dumps([asdict(s) for s in sample.structures])
        
        # device_groups needs careful handling to ensure nested objects are dicts
        # asdict recursively converts dataclasses to dicts
        device_groups_list = [asdict(dg) for dg in sample.device_groups]
        device_groups_json = json.dumps(device_groups_list)

        with get_db_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO samples (id, name, device_type, structures, device_groups, note, r_parasitic)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sample.id, sample.name, sample.device_type, structures_json, device_groups_json, sample.note, sample.r_parasitic)
            )

    def get_by_id(self, sample_id: str) -> Optional[Sample]:
        """
        Retrieves a sample by its ID.
        """
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_sample(row)

    def list_all(self) -> List[Sample]:
        """
        Lists all samples.
        """
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM samples ORDER BY created_at DESC").fetchall()
        conn.close()

        return [self._row_to_sample(row) for row in rows]

    def get_devices(self, sample_id: str) -> List[Device]:
        """
        Returns a flat list of devices for a given sample.
        Useful for listing all available devices without parsing groups manually.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            return []
        
        devices = []
        for group in sample.device_groups:
            devices.extend(group.devices)
        return devices

    def add_device_group(self, sample_id: str, device_group: DeviceGroup) -> None:
        """
        Adds a new DeviceGroup to an existing Sample.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample with ID {sample_id} not found")

        # Check for duplicate coord
        for dg in sample.device_groups:
            if dg.coord == device_group.coord:
                 # Logic for duplicate coord? maybe append? 
                 # For now, simplistic approach: raise error
                 raise ValueError(f"DeviceGroup with coord {device_group.coord} already exists")

        sample.device_groups.append(device_group)
        self._save_sample(sample)

    def add_device_to_group(self, sample_id: str, coord: tuple[int, int], device: Device) -> None:
        """
        Adds a device to an existing device group identified by coord.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample with ID {sample_id} not found")

        target_group = None
        for dg in sample.device_groups:
            # coord from DB might be list if not careful, but _row_to_sample handles conversion
            if dg.coord == coord:
                target_group = dg
                break
        
        if not target_group:
            raise ValueError(f"DeviceGroup with coord {coord} not found in sample {sample_id}")
            
        # Check for duplicate device ID globally in sample (or at least in group)
        # Global uniqueness check
        all_devs = [d.device_id for g in sample.device_groups for d in g.devices]
        if device.device_id in all_devs:
             raise ValueError(f"Device ID {device.device_id} already exists in sample {sample_id}")

        target_group.devices.append(device)
        self._save_sample(sample)

    def set_device_default_measurement(self, sample_id: str, device_id: str, measurement_type: str, measurement_id: str) -> None:
        """
        Updates the default measurement for a specific device.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        found = False
        for dg in sample.device_groups:
            for dev in dg.devices:
                if dev.device_id == device_id:
                    dev.default_measurements[measurement_type] = measurement_id
                    found = True
                    break
            if found: break
        
        if not found:
            raise ValueError(f"Device {device_id} not found in sample {sample_id}")

        self._save_sample(sample)

    def update_r_parasitic(self, sample_id: str, r_parasitic: float) -> None:
        """
        Update the r_parasitic value for a sample.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
        sample.r_parasitic = r_parasitic
        self._save_sample(sample)

    def remove_device_from_group(self, sample_id: str, device_id: str) -> None:
        """
        Remove a device from its group in the sample.
        """
        sample = self.get_by_id(sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")
            
        found = False
        for group in sample.device_groups:
            # Check if device is in this group
            original_count = len(group.devices)
            group.devices = [d for d in group.devices if d.device_id != device_id]
            if len(group.devices) < original_count:
                found = True
                break
        
        if found:
            self._save_sample(sample)
        else:
            raise ValueError(f"Device {device_id} not found in sample {sample_id}")

    def _row_to_sample(self, row) -> Sample:
        """
        Converts a DB row to a Sample dataclass instance.
        """
        structures_data = json.loads(row["structures"])
        structures = [LayerStructure(**s) for s in structures_data]

        device_groups_data = json.loads(row["device_groups"])
        device_groups = []
        for dg_data in device_groups_data:
            # Reconstruct Devices with backward compatibility for default_measurements
            devices = [
                Device(
                    device_id=d["device_id"],
                    area_um2=d["area_um2"],
                    note=d.get("note", ""),
                    default_measurements=d.get("default_measurements", {})
                )
                for d in dg_data["devices"]
            ]
            
            # Reconstruct DeviceGroup
            # coord in JSON is list main, tuple in python
            dg = DeviceGroup(
                coord=tuple(dg_data["coord"]), 
                thick_nm=dg_data["thick_nm"],
                shared_properties=dg_data["shared_properties"],
                devices=devices
            )
            device_groups.append(dg)

        # r_parasitic may not exist in older DBs
        r_parasitic = None
        try:
            r_parasitic = row["r_parasitic"]
        except (IndexError, KeyError):
            pass

        return Sample(
            id=row["id"],
            name=row["name"],
            device_type=row["device_type"],
            structures=structures,
            device_groups=device_groups,
            note=row["note"],
            r_parasitic=r_parasitic
        )
