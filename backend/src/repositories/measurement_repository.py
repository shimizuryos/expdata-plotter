import json
from typing import Optional, List
from dataclasses import asdict
from ..database import get_db_connection, get_db_cursor
from ..models.db_models import Measurement

class MeasurementRepository:
    def insert(self, measurement: Measurement) -> None:
        """
        Inserts a measurement record.
        """
        metadata_json = json.dumps(measurement.metadata)
        data_json = json.dumps(measurement.data) if measurement.data else None
        derived_json = json.dumps(measurement.derived) if measurement.derived else None

        with get_db_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO measurements 
                (id, sample_id, device_id, measurement_type, metadata, data, derived, file_ref, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement.id,
                    measurement.sample_id,
                    measurement.device_id,
                    measurement.measurement_type,
                    metadata_json,
                    data_json,
                    derived_json,
                    measurement.file_ref,
                    measurement.measured_at
                )
            )

    def get_by_id(self, measurement_id: str) -> Optional[Measurement]:
        """
        Retrieves a measurement by ID.
        """
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM measurements WHERE id = ?", (measurement_id,)).fetchone()
        conn.close()

        if row is None:
            return None
        return self._row_to_measurement(row)

    def find_by_device(self, sample_id: str, device_id: str, measurement_type: Optional[str] = None) -> List[Measurement]:
        """
        Finds measurements for a specific device.
        Optionally filters by measurement type.
        """
        query = "SELECT * FROM measurements WHERE sample_id = ? AND device_id = ?"
        params = [sample_id, device_id]

        if measurement_type:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
            
        query += " ORDER BY measured_at DESC"

        conn = get_db_connection()
        rows = conn.execute(query, tuple(params)).fetchall()
        conn.close()

        return [self._row_to_measurement(row) for row in rows]

    def find_by_sample(self, sample_id: str, measurement_type: Optional[str] = None) -> List[Measurement]:
        """
        Finds measurements for a whole sample.
        """
        query = "SELECT * FROM measurements WHERE sample_id = ?"
        params = [sample_id]

        if measurement_type:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
        
        query += " ORDER BY measured_at DESC"

        conn = get_db_connection()
        rows = conn.execute(query, tuple(params)).fetchall()
        conn.close()

        return [self._row_to_measurement(row) for row in rows]

    def delete(self, measurement_id: str) -> None:
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM measurements WHERE id = ?", (measurement_id,))
            conn.commit()
        finally:
            conn.close()

    def _row_to_measurement(self, row) -> Measurement:
        """
        Converts a DB row to a Measurement object.
        """
        return Measurement(
            id=row["id"],
            sample_id=row["sample_id"],
            device_id=row["device_id"],
            measurement_type=row["measurement_type"],
            metadata=json.loads(row["metadata"]),
            data=json.loads(row["data"]) if row["data"] else None,
            derived=json.loads(row["derived"]) if row["derived"] else None,
            file_ref=row["file_ref"],
            measured_at=row["measured_at"]
        )
