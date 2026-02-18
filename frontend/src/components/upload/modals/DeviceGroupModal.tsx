import React, { useState } from "react";

interface DeviceGroupModalProps {
    isOpen: boolean;
    onClose: () => void;
    onCreate: (groupData: any) => Promise<void>;
}

export function DeviceGroupModal({ isOpen, onClose, onCreate }: DeviceGroupModalProps) {
    const [newGroup, setNewGroup] = useState({
        coordX: 0,
        coordY: 0,
        thickNmStr: '{"MgO": 2.0}',
        sharedPropsStr: '{"Ra_parasitic": 0}',
        groupClass: "area_variation",
        variations: [] as { suffix: string, area_um2: number }[],
        singleArea: 0
    });
    const [newVariation, setNewVariation] = useState({ suffix: "", area_um2: 10 });

    if (!isOpen) return null;

    const handleSubmit = async () => {
        const variations = newGroup.groupClass === "area_variation"
            ? newGroup.variations
            : [{ suffix: "", area_um2: newGroup.singleArea }];

        await onCreate({
            coord: [newGroup.coordX, newGroup.coordY],
            thick_nm: JSON.parse(newGroup.thickNmStr || "{}"),
            shared_properties: JSON.parse(newGroup.sharedPropsStr || "{}"),
            group_class: newGroup.groupClass,
            variations: variations
        });

        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto shadow-xl">
                <h2 className="text-2xl font-bold mb-4">Create Device Group</h2>
                <div className="space-y-4">
                    <div className="flex gap-4">
                        <div className="w-1/2">
                            <label className="text-xs font-bold text-gray-500">X Coordinate</label>
                            <input type="number" className="w-full p-2 border rounded" placeholder="X" value={newGroup.coordX} onChange={e => setNewGroup({ ...newGroup, coordX: parseInt(e.target.value) || 0 })} />
                        </div>
                        <div className="w-1/2">
                            <label className="text-xs font-bold text-gray-500">Y Coordinate</label>
                            <input type="number" className="w-full p-2 border rounded" placeholder="Y" value={newGroup.coordY} onChange={e => setNewGroup({ ...newGroup, coordY: parseInt(e.target.value) || 0 })} />
                        </div>
                    </div>

                    <div>
                        <label className="text-sm font-semibold">Group Class</label>
                        <select className="w-full p-2 border rounded" value={newGroup.groupClass} onChange={e => setNewGroup({ ...newGroup, groupClass: e.target.value })}>
                            <option value="area_variation">Area Variation (Multiple Devices)</option>
                            <option value="single">Single Device</option>
                        </select>
                    </div>

                    {newGroup.groupClass === "area_variation" ? (
                        <div className="border p-4 rounded bg-gray-50">
                            <h3 className="font-semibold mb-2 text-sm">Variations (Suffix + Area)</h3>
                            {newGroup.variations.map((v, i) => (
                                <div key={i} className="text-sm bg-white border p-2 mb-1 flex justify-between rounded">
                                    <span>Suffix: {v.suffix}, Area: {v.area_um2} um²</span>
                                </div>
                            ))}
                            <div className="flex gap-2 mt-2">
                                <input className="p-2 border rounded w-1/3 text-sm" placeholder="Suffix (c)" value={newVariation.suffix} onChange={e => setNewVariation({ ...newVariation, suffix: e.target.value })} />
                                <input type="number" className="p-2 border rounded w-1/3 text-sm" placeholder="Area (250)" value={newVariation.area_um2} onChange={e => setNewVariation({ ...newVariation, area_um2: parseFloat(e.target.value) })} />
                                <button onClick={() => {
                                    if (!newVariation.suffix) return;
                                    setNewGroup({ ...newGroup, variations: [...newGroup.variations, newVariation] });
                                    setNewVariation({ suffix: "", area_um2: 10 });
                                }} className="bg-blue-500 text-white px-3 rounded text-sm">Add</button>
                            </div>
                        </div>
                    ) : (
                        <div>
                            <label className="text-sm font-semibold">Area (um²)</label>
                            <input type="number" className="w-full p-2 border rounded" value={newGroup.singleArea} onChange={e => setNewGroup({ ...newGroup, singleArea: parseFloat(e.target.value) })} />
                        </div>
                    )}

                    <div>
                        <label className="text-sm font-semibold">Thickness (JSON)</label>
                        <input className="w-full p-2 border rounded font-mono text-sm" value={newGroup.thickNmStr} onChange={e => setNewGroup({ ...newGroup, thickNmStr: e.target.value })} />
                    </div>
                    <div>
                        <label className="text-sm font-semibold">Shared Properties (JSON)</label>
                        <input className="w-full p-2 border rounded font-mono text-sm" value={newGroup.sharedPropsStr} onChange={e => setNewGroup({ ...newGroup, sharedPropsStr: e.target.value })} />
                    </div>
                </div>
                <div className="flex justify-end gap-2 mt-6">
                    <button onClick={onClose} className="px-4 py-2 border rounded hover:bg-gray-100">Cancel</button>
                    <button onClick={handleSubmit} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Create</button>
                </div>
            </div>
        </div>
    );
}
