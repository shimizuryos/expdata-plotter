import React, { useState } from "react";

interface AppendVariationsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onAppend: (variations: any[]) => Promise<void>;
    selectedCoord: string; // JSON string or display string
}

export function AppendVariationsModal({ isOpen, onClose, onAppend, selectedCoord }: AppendVariationsModalProps) {
    const [appendVariations, setAppendVariations] = useState([] as { suffix: string, area_um2: number }[]);
    const [newAppendVariation, setNewAppendVariation] = useState({ suffix: "", area_um2: 10 });

    if (!isOpen) return null;

    const handleSubmit = async () => {
        await onAppend(appendVariations);
        onClose();
        setAppendVariations([]);
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto shadow-xl">
                <h2 className="text-2xl font-bold mb-4">Add Variations</h2>
                <p className="mb-4 text-sm text-gray-600">Appending to group at {selectedCoord}</p>

                <div className="border p-4 rounded bg-gray-50 mb-4">
                    <h3 className="font-semibold mb-2 text-sm">New Variations</h3>
                    {appendVariations.map((v, i) => (
                        <div key={i} className="text-sm bg-white border p-2 mb-1 flex justify-between rounded">
                            <span>Suffix: {v.suffix}, Area: {v.area_um2} um²</span>
                        </div>
                    ))}
                    <div className="flex gap-2 mt-2">
                        <input className="p-2 border rounded w-1/3 text-sm" placeholder="Suffix (c)" value={newAppendVariation.suffix} onChange={e => setNewAppendVariation({ ...newAppendVariation, suffix: e.target.value })} />
                        <input type="number" className="p-2 border rounded w-1/3 text-sm" placeholder="Area" value={newAppendVariation.area_um2} onChange={e => setNewAppendVariation({ ...newAppendVariation, area_um2: parseFloat(e.target.value) })} />
                        <button onClick={() => {
                            if (!newAppendVariation.suffix) return;
                            setAppendVariations([...appendVariations, newAppendVariation]);
                            setNewAppendVariation({ suffix: "", area_um2: 10 });
                        }} className="bg-blue-500 text-white px-3 rounded text-sm">Add</button>
                    </div>
                </div>

                <div className="flex justify-end gap-2 mt-6">
                    <button onClick={onClose} className="px-4 py-2 border rounded hover:bg-gray-100">Cancel</button>
                    <button onClick={handleSubmit} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Append</button>
                </div>
            </div>
        </div>
    );
}
