"use client";

import { useEffect, useState } from "react";
import DynamicPlot from "@/components/DynamicPlot";
import Link from "next/link";

export default function HanlePlotPage() {
    const [plotData, setPlotData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("http://localhost:8000/api/plots/hanle")
            .then((res) => res.json())
            .then((data) => {
                setPlotData(data);
                setLoading(false);
            });
    }, []);

    return (
        <div className="flex flex-col items-center min-h-screen p-8">
            <div className="w-full max-w-5xl flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold">Hanle Effect</h1>
                <Link href="/" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                    Back to Home
                </Link>
            </div>

            <div className="w-full max-w-6xl bg-white p-4 rounded-xl shadow-lg border border-gray-200 min-h-[600px] flex items-center justify-center">
                {loading && <p>Loading data...</p>}
                {!loading && plotData && (
                    <DynamicPlot
                        data={plotData.data}
                        layout={{ ...plotData.layout, autosize: true, height: 600 }}
                        useResizeHandler={true}
                        style={{ width: "100%", height: "100%" }}
                    />
                )}
            </div>
        </div>
    );
}
