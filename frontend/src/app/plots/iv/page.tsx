"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

const IVSingleTab = dynamic(() => import("@/components/plots/IVSingleTab"), { ssr: false });
const IVSimulTab = dynamic(() => import("@/components/plots/IVSimulTab"), { ssr: false });
const IVGroupTab = dynamic(() => import("@/components/plots/IVGroupTab"), { ssr: false });

const API = "http://localhost:8000/api/data";

type TabKey = "single" | "simultaneous" | "group";

export default function IvPlotPage() {
    const [activeTab, setActiveTab] = useState<TabKey>("single");
    const [samples, setSamples] = useState<any[]>([]);

    useEffect(() => {
        fetch(`${API}/samples`).then(r => r.json()).then(setSamples);
    }, []);

    const tabs: { key: TabKey; label: string }[] = [
        { key: "single", label: "Single Plot" },
        { key: "simultaneous", label: "Simultaneous Plot" },
        { key: "group", label: "Group Plot" },
    ];

    return (
        <div className="flex flex-col items-center min-h-screen p-8 bg-gray-50">
            <div className="w-full max-w-7xl flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold">IV Characteristics</h1>
                <Link href="/" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                    Back to Home
                </Link>
            </div>

            {/* Tabs */}
            <div className="w-full max-w-7xl mb-4">
                <div className="flex border-b border-gray-300">
                    {tabs.map(t => (
                        <button
                            key={t.key}
                            onClick={() => setActiveTab(t.key)}
                            className={`px-6 py-3 text-sm font-semibold transition-colors
                                ${activeTab === t.key
                                    ? "border-b-2 border-blue-600 text-blue-600 bg-white"
                                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                                }`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Tab content */}
            <div className="w-full max-w-7xl bg-white p-6 rounded-xl shadow-lg border border-gray-200 min-h-[600px]">
                {activeTab === "single" && <IVSingleTab samples={samples} />}
                {activeTab === "simultaneous" && <IVSimulTab samples={samples} />}
                {activeTab === "group" && <IVGroupTab samples={samples} />}
            </div>
        </div>
    );
}
