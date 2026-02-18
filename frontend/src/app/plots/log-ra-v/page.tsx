"use client";

import { useEffect, useState } from "react";
import DynamicPlot from "@/components/DynamicPlot";
import Link from "next/link";

export default function LogRaVPlotPage() {
    const [plotData, setPlotData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [description, setDescription] = useState<string>("");

    // Hardcoded ID for now, as per current requirement.
    // In future, this could be a dynamic route [id].
    const plotId = "ra_v_plot_data_4K";

    useEffect(() => {
        fetch(`http://localhost:8000/api/plot/${plotId}`)
            .then(async (res) => {
                if (!res.ok) {
                    const text = await res.text();
                    throw new Error(`Failed to fetch data: ${res.status} ${text}`);
                }
                return res.json();
            })
            .then((data) => {
                // API returns { plot: {...}, description: "...", plot_type: "..." }
                if (data.plot) {
                    setPlotData(data.plot);
                    setDescription(data.description || "");
                } else {
                    // Fallback if structure is different
                    setPlotData(data);
                }
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setError(err.message);
                setLoading(false);
            });
    }, [plotId]);

    return (
        <div className="flex flex-col items-center min-h-screen p-8">
            <div className="w-full max-w-6xl flex justify-between items-center mb-8">
                <div className="flex flex-col">
                    <h1 className="text-3xl font-bold">Log RA vs V Plot</h1>
                    {description && <p className="text-gray-500 mt-1">{description}</p>}
                </div>
                <Link href="/plots/systematic" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                    Back to Systematic Plots
                </Link>
            </div>

            <div className="w-full max-w-6xl bg-white p-4 rounded-xl shadow-lg border border-gray-200 min-h-[600px] flex items-center justify-center">
                {loading && <p>Loading data...</p>}
                {error && <p className="text-red-500">Error: {error}</p>}

                {!loading && !error && plotData && (
                    <DynamicPlot
                        data={plotData.data}
                        layout={{
                            ...plotData.layout,
                            autosize: true,
                            width: undefined, // Let it fill container
                            height: 700
                        }}
                        useResizeHandler={true}
                        style={{ width: "100%", height: "100%" }}
                    />
                )}
            </div>
        </div>
    );
}
