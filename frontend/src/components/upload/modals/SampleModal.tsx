import React, { useState } from "react";
import { parseThicknessSequence } from "@/utils/unitParser";

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
        structures: [] as { material: string, thick_nm_variation: string }[],
        max_x: 22,
        max_y: 22
    });
    const [newStructure, setNewStructure] = useState({ material: "", variationStr: "" });
    const [editIndex, setEditIndex] = useState<number | null>(null);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleSubmit = async () => {
        try {
            setErrorMsg(null);

            const structures = newSample.structures.map(s => {
                let parsedVars: number[] | null = null;
                if (s.thick_nm_variation) {
                    parsedVars = parseThicknessSequence(s.thick_nm_variation);
                }

                return {
                    material: s.material,
                    thick_nm_variation: parsedVars
                };
            });

            await onCreate({
                id: newSample.id,
                name: newSample.name,
                device_type: "three_terminal_hanle",
                structures: structures,
                note: newSample.note,
                max_x: newSample.max_x,
                max_y: newSample.max_y
            });

            // Reset form
            setNewSample({ id: "", name: "", note: "", structures: [], max_x: 22, max_y: 22 });
            setEditIndex(null);
            setErrorMsg(null);
        } catch (e: any) {
            setErrorMsg(e.message || "An error occurred parsing the input.");
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white p-6 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-xl">
                <h2 className="text-2xl font-bold mb-4">Create New Sample</h2>

                {errorMsg && (
                    <div className="mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                        <strong className="font-bold">Validation Error! </strong>
                        <span className="block sm:inline">{errorMsg}</span>
                    </div>
                )}

                <div className="space-y-4">
                    <input className="w-full p-2 border rounded" placeholder="Sample ID (e.g. 250918_cofe)" value={newSample.id} onChange={e => setNewSample({ ...newSample, id: e.target.value })} />
                    <input className="w-full p-2 border rounded" placeholder="Sample Name" value={newSample.name} onChange={e => setNewSample({ ...newSample, name: e.target.value })} />

                    <div className="flex gap-4">
                        <div className="flex-1">
                            <label className="block text-sm text-gray-600 mb-1">Grid Max X</label>
                            <input type="number" className="w-full p-2 border rounded" value={newSample.max_x} onChange={e => setNewSample({ ...newSample, max_x: parseInt(e.target.value) || 22 })} />
                        </div>
                        <div className="flex-1">
                            <label className="block text-sm text-gray-600 mb-1">Grid Max Y</label>
                            <input type="number" className="w-full p-2 border rounded" value={newSample.max_y} onChange={e => setNewSample({ ...newSample, max_y: parseInt(e.target.value) || 22 })} />
                        </div>
                    </div>

                    <textarea className="w-full p-2 border rounded" placeholder="Note" value={newSample.note} onChange={e => setNewSample({ ...newSample, note: e.target.value })} />

                    <div className="border p-4 rounded bg-gray-50">
                        <h3 className="font-semibold mb-2 text-sm">Layers</h3>
                        {newSample.structures.map((s, i) => (
                            <div key={i} className="text-sm bg-white border p-2 mb-1 flex justify-between items-center rounded">
                                <span><span className="font-medium">{s.material}</span> {s.thick_nm_variation ? <span className="text-gray-500">({s.thick_nm_variation})</span> : <span className="text-gray-400 italic">No variations</span>}</span>
                                <div className="flex gap-3">
                                    <button onClick={() => {
                                        setNewStructure({ material: s.material, variationStr: s.thick_nm_variation });
                                        setEditIndex(i);
                                    }} className="text-blue-500 hover:text-blue-700">Edit</button>
                                    <button onClick={() => {
                                        setNewSample({ ...newSample, structures: newSample.structures.filter((_, idx) => idx !== i) });
                                        if (editIndex === i) {
                                            setNewStructure({ material: "", variationStr: "" });
                                            setEditIndex(null);
                                        }
                                    }} className="text-red-500 hover:text-red-700">Delete</button>
                                </div>
                            </div>
                        ))}
                        <div className="flex flex-col gap-1 mt-3 pt-3 border-t">
                            <div className="flex flex-wrap sm:flex-nowrap gap-2 items-center">
                                <input className="p-2 border rounded flex-auto min-w-[120px] text-sm" placeholder="Material (e.g. MgO)" value={newStructure.material} onChange={e => setNewStructure({ ...newStructure, material: e.target.value })} />
                                <input className="p-2 border rounded flex-auto min-w-[200px] text-sm" placeholder="Thickness (e.g. 1.0, 2.0 nm)" value={newStructure.variationStr} onChange={e => setNewStructure({ ...newStructure, variationStr: e.target.value })} />
                                <button onClick={() => {
                                    if (!newStructure.material) return;

                                    const updatedStructures = [...newSample.structures];
                                    if (editIndex !== null) {
                                        updatedStructures[editIndex] = { material: newStructure.material, thick_nm_variation: newStructure.variationStr };
                                    } else {
                                        updatedStructures.push({ material: newStructure.material, thick_nm_variation: newStructure.variationStr });
                                    }

                                    setNewSample({ ...newSample, structures: updatedStructures });
                                    setNewStructure({ material: "", variationStr: "" });
                                    setEditIndex(null);
                                }} className="bg-blue-500 text-white px-3 py-2 rounded text-sm hover:bg-blue-600 transition whitespace-nowrap shrink-0">
                                    {editIndex !== null ? "Update Layer" : "Add Layer"}
                                </button>
                            </div>
                            <p className="text-xs text-gray-500">For thickness variations, enter values separated by commas (e.g. "1.0, 1.5, 2.0"). Leave empty if there is no variation.</p>
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
