import React, { useState } from "react";
import { ItemList } from "../shared/ItemList";

interface MeasurementSectionProps {
    sampleId: string;
    deviceId: string;
    existingMeasurements: any[];
    onRegister: (data: any) => Promise<void>;
    onDeleteMeasurement: (id: string) => Promise<void>;
    message: string | null;
}

export function MeasurementSection({
    sampleId, deviceId, existingMeasurements,
    onRegister, onDeleteMeasurement, message
}: MeasurementSectionProps) {

    const [measurementType, setMeasurementType] = useState<string>("IV");
    const [metadataStr, setMetadataStr] = useState<string>('{"temp_K": 300}');
    const [fileRef, setFileRef] = useState<string>("");
    const [setAsDefault, setSetAsDefault] = useState<boolean>(false);
    const [hanleDerivedStr, setHanleDerivedStr] = useState<string>('{"ps_percent": null, "ra_ohm_um2": null, "rms": null}');
    const [isSubmitting, setIsSubmitting] = useState(false);

    if (!deviceId) {
        return <div className="opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold mb-4">4. Measurement Info</h2>
        </div>;
    }

    const handleBrowse = async () => {
        try {
            const res = await fetch("http://localhost:8000/api/utils/browse-file", { method: "POST" });
            if (res.ok) {
                const data = await res.json();
                if (data.path) setFileRef(data.path);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleRegisterClick = async () => {
        setIsSubmitting(true);
        try {
            const metadata = JSON.parse(metadataStr);
            let derived = null;
            if (measurementType === "Hanle") {
                derived = JSON.parse(hanleDerivedStr);
            }

            await onRegister({
                sample_id: sampleId,
                device_id: deviceId,
                measurement_type: measurementType,
                metadata: metadata,
                file_ref: fileRef,
                derived: derived,
                set_as_default: setAsDefault
            });
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div>
            <h2 className="text-xl font-semibold mb-4">4. Measurement Info</h2>

            <ItemList
                title="Existing Measurements"
                items={existingMeasurements}
                renderItem={(m: any) => `${m.measurement_type} (${m.measured_at?.substring(0, 10)}) - ${m.file_ref?.split('/').pop()}`}
                onDelete={(m: any) => onDeleteMeasurement(m.id)}
            />

            <div className="grid grid-cols-2 gap-4 mb-4 mt-4">
                <div>
                    <label className="block text-sm font-medium">Type</label>
                    <select className="w-full p-2 border rounded" value={measurementType} onChange={(e) => setMeasurementType(e.target.value)}>
                        <option value="IV">IV</option>
                        <option value="Hanle">Hanle</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-medium">Metadata (JSON)</label>
                    <input className="w-full p-2 border rounded font-mono text-sm" value={metadataStr} onChange={(e) => setMetadataStr(e.target.value)} />
                </div>
            </div>

            {measurementType === "Hanle" && (
                <div className="mb-4">
                    <label className="block text-sm font-medium">Hanle Derived (JSON) [ps, ra, rms]</label>
                    <input className="w-full p-2 border rounded font-mono text-sm" value={hanleDerivedStr} onChange={(e) => setHanleDerivedStr(e.target.value)} />
                </div>
            )}

            <div className="mb-4">
                <label className="block text-sm font-medium">File Path</label>
                <div className="flex gap-2">
                    <input className="w-full p-2 border rounded" placeholder="/path/to/data" value={fileRef} onChange={(e) => setFileRef(e.target.value)} />
                    <button onClick={handleBrowse} className="bg-gray-200 hover:bg-gray-300 px-4 rounded text-sm whitespace-nowrap">Browse...</button>
                </div>
                <p className="text-xs text-gray-500 mt-1">Accepts drag & drop of text path. Use "Browse" for file dialog.</p>
            </div>

            <div className="flex items-center space-x-2 mb-4">
                <input type="checkbox" checked={setAsDefault} onChange={(e) => setSetAsDefault(e.target.checked)} />
                <label className="text-sm">Set as default</label>
            </div>

            <button
                onClick={handleRegisterClick}
                className={`w-full text-white font-bold py-3 rounded ${isSubmitting ? 'bg-blue-400 cursor-wait' : 'bg-blue-600 hover:bg-blue-700'}`}
                disabled={isSubmitting}
            >
                Register Measurement
            </button>

            {message && <div className={`mt-4 p-4 rounded ${message.includes("Error") ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}>{message}</div>}
        </div>
    );
}
