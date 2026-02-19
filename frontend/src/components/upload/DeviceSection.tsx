import React from "react";

interface DeviceSectionProps {
    selectedCoord: string;
    devicesInGroup: any[];
    selectedDeviceId: string;
    onSelect: (id: string) => void;
}

export function DeviceSection({ selectedCoord, devicesInGroup, selectedDeviceId, onSelect }: DeviceSectionProps) {
    if (!selectedCoord) {
        return <div className="border-b pb-6 opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold mb-4">3. Select Device</h2>
            <select className="w-full p-2 border rounded" disabled><option>-- Select Device --</option></select>
        </div>;
    }

    return (
        <div className="border-b pb-6">
            <h2 className="text-xl font-semibold mb-4">3. Select Device</h2>
            <select
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 outline-none"
                value={selectedDeviceId}
                onChange={(e) => onSelect(e.target.value)}
            >
                <option value="">-- Select Device --</option>
                {devicesInGroup.map((d: any) => (
                    <option key={d.device_id} value={d.device_id}>{d.device_id} ({d.area_um2} um²)</option>
                ))}
            </select>
        </div>
    );
}
