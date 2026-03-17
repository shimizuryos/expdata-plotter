import React, { useState } from "react";
import { DeviceGroupModal } from "./modals/DeviceGroupModal";
import { AppendVariationsModal } from "./modals/AppendVariationsModal";
import { ItemList } from "../shared/ItemList";

interface DeviceGroupSectionProps {
    sampleId: string;
    selectedCoord: string;
    deviceGroups: any[];
    devicesInGroup: any[];
    onSelect: (coord: string) => void;
    onRefreshGroups: () => Promise<void>;
    onDeleteDevice: (deviceId: string) => Promise<void>;
}

export function DeviceGroupSection({
    sampleId, selectedCoord, deviceGroups, devicesInGroup,
    onSelect, onRefreshGroups, onDeleteDevice
}: DeviceGroupSectionProps) {
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isAppendModalOpen, setIsAppendModalOpen] = useState(false);

    if (!sampleId) {
        return <div className="border-b pb-6 opacity-50 pointer-events-none">
            <h2 className="text-xl font-semibold mb-4">2. Select Device Group</h2>
            <select className="w-full p-2 border rounded" disabled><option>-- Select Group --</option></select>
        </div>;
    }

    const handleCreate = async (data: any) => {
        const res = await fetch(`http://localhost:8000/api/data/samples/${sampleId}/device-groups`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        if (!res.ok) {
            const err = await res.json();
            const errMsg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
            throw new Error(errMsg);
        }

        await onRefreshGroups();
    };

    const handleAppend = async (variations: any[]) => {
        const coord = JSON.parse(selectedCoord);
        const res = await fetch(`http://localhost:8000/api/data/samples/${sampleId}/device-groups/append`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ coord, variations })
        });
        if (!res.ok) {
            const err = await res.json();
            const errMsg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
            throw new Error(errMsg);
        }

        await onRefreshGroups();
        // Trigger refresh of devices list is handled by parent or effect there?
        // Actually, if we refresh groups, the parent `deviceGroups` updates, but `devicesInGroup` state might need manual update if it's separate.
        // We will assume `onRefreshGroups` updates parent state which triggers re-render.
        // But `devicesInGroup` in parent is derived from `selectedCoord` + `deviceGroups`. Assumed parent handles this.
    };

    return (
        <div className="border-b pb-6">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">2. Select Device Group</h2>
                <div className="flex gap-2">
                    <button onClick={() => setIsCreateModalOpen(true)} className="text-sm bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 transition">+ New Group</button>
                    {selectedCoord && (
                        <button onClick={() => setIsAppendModalOpen(true)} className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition">+ Add Variations</button>
                    )}
                </div>
            </div>

            <select
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 outline-none mb-2"
                value={selectedCoord}
                onChange={(e) => onSelect(e.target.value)}
            >
                <option value="">-- Select Group (Coord) --</option>
                {deviceGroups.map(g => (
                    <option key={JSON.stringify(g.coord)} value={JSON.stringify(g.coord)}>
                        Coord: {JSON.stringify(g.coord)} - {g.devices.length} devices
                    </option>
                ))}
            </select>

            <ItemList
                title="Devices in Group"
                items={devicesInGroup}
                renderItem={(d: any) => `${d.device_id} (${d.area_um2} um²)`}
                onDelete={(d: any) => onDeleteDevice(d.device_id)}
            />

            <DeviceGroupModal
                isOpen={isCreateModalOpen}
                onClose={() => setIsCreateModalOpen(false)}
                onCreate={handleCreate}
            />

            <AppendVariationsModal
                isOpen={isAppendModalOpen}
                onClose={() => setIsAppendModalOpen(false)}
                onAppend={handleAppend}
                selectedCoord={selectedCoord}
            />
        </div>
    );
}
