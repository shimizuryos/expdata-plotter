import React, { useState } from "react";
import { SampleModal } from "./modals/SampleModal";

interface SampleSectionProps {
    samples: any[];
    selectedSampleId: string;
    onSelect: (id: string) => void;
    onRefresh: () => Promise<void>;
}

export function SampleSection({ samples, selectedSampleId, onSelect, onRefresh }: SampleSectionProps) {
    const [isModalOpen, setIsModalOpen] = useState(false);

    const handleCreate = async (data: any) => {
        const res = await fetch("http://localhost:8000/api/data/samples", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail);
        }

        await onRefresh();
        onSelect(data.id);
        setIsModalOpen(false);
    };

    return (
        <div className="border-b pb-6">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">1. Select Sample</h2>
                <button onClick={() => setIsModalOpen(true)} className="text-sm bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 transition">+ New Sample</button>
            </div>
            <select
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 outline-none"
                value={selectedSampleId}
                onChange={(e) => onSelect(e.target.value)}
            >
                <option value="">-- Select Sample --</option>
                {samples.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
            </select>

            <SampleModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onCreate={handleCreate}
            />
        </div>
    );
}
