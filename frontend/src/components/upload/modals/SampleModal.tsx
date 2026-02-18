import React, { useState } from "react";

interface SampleModalProps {
    isOpen: boolean;
    onClose: () => void;
    onCreate: (sample: any) => Promise<void>;
}

export function SampleModal({ isOpen, onClose, onCreate }: SampleModalProps) {
    const [newSample, setNewSample] = useState({
        id: "",
        name: "",
        note: "",
        structures: [] as { material: string, thick_nm_variation: string }[]
    });
    const [newStructure, setNewStructure] = useState({ material: "", variationStr: "" });

    if (!isOpen) return null;

    const handleSubmit = async () => {
        const structures = newSample.structures.map(s => ({
            material: s.material,
            thick_nm_variation: s.thick_nm_variation ? s.thick_nm_variation.split(",").map(v => parseFloat(v.trim())) : null
        }));

        await onCreate({
            id: newSample.id,
            name: newSample.name,
            device_type: "three_terminal_hanle",
            structures: structures,
            note: newSample.note
        });

        // Reset form
        setNewSample({ id: "", name: "", note: "", structures: [] });
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto shadow-xl">
                <h2 className="text-2xl font-bold mb-4">Create New Sample</h2>
                <div className="space-y-4">
                    <input className="w-full p-2 border rounded" placeholder="Sample ID (e.g. 250918_cofe)" value={newSample.id} onChange={e => setNewSample({ ...newSample, id: e.target.value })} />
                    <input className="w-full p-2 border rounded" placeholder="Sample Name" value={newSample.name} onChange={e => setNewSample({ ...newSample, name: e.target.value })} />
                    <textarea className="w-full p-2 border rounded" placeholder="Note" value={newSample.note} onChange={e => setNewSample({ ...newSample, note: e.target.value })} />

                    <div className="border p-4 rounded bg-gray-50">
                        <h3 className="font-semibold mb-2 text-sm">Layers</h3>
                        {newSample.structures.map((s, i) => (
                            <div key={i} className="text-sm bg-white border p-2 mb-1 flex justify-between rounded">
                                <span>{s.material} {s.thick_nm_variation ? `(${s.thick_nm_variation})` : ""}</span>
                            </div>
                        ))}
                        <div className="flex gap-2 mt-2">
                            <input className="p-2 border rounded flex-1 text-sm" placeholder="Material (e.g. MgO)" value={newStructure.material} onChange={e => setNewStructure({ ...newStructure, material: e.target.value })} />
                            <input className="p-2 border rounded flex-1 text-sm" placeholder="Variations (1.0, 2.0)" value={newStructure.variationStr} onChange={e => setNewStructure({ ...newStructure, variationStr: e.target.value })} />
                            <button onClick={() => {
                                if (!newStructure.material) return;
                                setNewSample({ ...newSample, structures: [...newSample.structures, { material: newStructure.material, thick_nm_variation: newStructure.variationStr }] });
                                setNewStructure({ material: "", variationStr: "" });
                            }} className="bg-blue-500 text-white px-3 rounded text-sm">Add</button>
                        </div>
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
