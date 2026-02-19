"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

const DynamicPlot = dynamic(() => import("@/components/DynamicPlot"), { ssr: false });

const API = "http://localhost:8000/api/data";

type Metric = "ps_percent" | "ra_ohm_um2" | "rms";

const metricLabels: Record<Metric, string> = {
    ps_percent: "Ps (%)",
    ra_ohm_um2: "RA (Ω·μm²)",
    rms: "RMS",
};

export default function HeatmapPage() {
    const [samples, setSamples] = useState<any[]>([]);
    const [sampleId, setSampleId] = useState("");
    const [metric, setMetric] = useState<Metric>("ps_percent");
    const [heatmapData, setHeatmapData] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetch(`${API}/samples`).then(r => r.json()).then(setSamples);
    }, []);

    useEffect(() => {
        if (!sampleId) return;
        setLoading(true);
        fetch(`${API}/samples/${sampleId}/heatmap-data`)
            .then(r => r.json())
            .then(data => { setHeatmapData(data); setLoading(false); })
            .catch(() => setLoading(false));
    }, [sampleId]);

    const buildPlotData = () => {
        if (!heatmapData) return null;

        // Group by unique coord, pick representative value (e.g., largest area device)
        const coordMap = new Map<string, { x: number; y: number; val: number | null; label: string }>();

        for (const d of heatmapData.devices) {
            const key = `${d.coord[0]},${d.coord[1]}`;
            const val = d.hanle?.[metric] ?? null;
            // Keep the first device with a value for each coord (or update to largest area)
            if (!coordMap.has(key) || (val !== null && coordMap.get(key)!.val === null)) {
                coordMap.set(key, { x: d.coord[0], y: d.coord[1], val, label: d.device_id });
            }
        }

        const entries = Array.from(coordMap.values()).filter(e => e.val !== null);
        if (entries.length === 0) return null;

        // Build scatter-based heatmap (colored markers with text)
        const xs = entries.map(e => e.x);
        const ys = entries.map(e => e.y);
        const vals = entries.map(e => e.val!);
        const labels = entries.map(e => e.label);
        const textLabels = vals.map(v => v.toFixed(2));

        return {
            data: [{
                type: "scatter" as const,
                mode: "text+markers" as const,
                x: xs,
                y: ys,
                text: textLabels,
                textposition: "top center" as const,
                textfont: { size: 10 },
                marker: {
                    size: 30,
                    color: vals,
                    colorscale: "Viridis",
                    showscale: true,
                    colorbar: { title: metricLabels[metric] },
                },
                customdata: labels,
                hovertemplate: "<b>%{customdata}</b><br>(%{x}, %{y})<br>" + metricLabels[metric] + ": %{marker.color:.3f}<extra></extra>",
            }],
            layout: {
                title: `${metricLabels[metric]} Heatmap`,
                xaxis: { title: "X", showgrid: true, gridcolor: "LightGray", dtick: 1 },
                yaxis: { title: "Y", autorange: "reversed" as const, showgrid: true, gridcolor: "LightGray", dtick: 1 },
                plot_bgcolor: "white",
                height: 700,
            },
        };
    };

    const plotConfig = buildPlotData();

    return (
        <div className="flex flex-col items-center min-h-screen p-8 bg-gray-50">
            <div className="w-full max-w-6xl flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold">Heatmap Visualization</h1>
                <Link href="/plots/systematic" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                    Back to Systematic Plots
                </Link>
            </div>

            <div className="w-full max-w-6xl bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                        <label className="block text-sm font-semibold mb-1">Sample</label>
                        <select className="w-full p-2 border rounded" value={sampleId} onChange={e => setSampleId(e.target.value)}>
                            <option value="">-- Select --</option>
                            {samples.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm font-semibold mb-1">Metric</label>
                        <div className="flex gap-2">
                            {(Object.keys(metricLabels) as Metric[]).map(m => (
                                <button key={m} onClick={() => setMetric(m)}
                                    className={`px-4 py-2 rounded text-sm font-medium transition ${metric === m
                                        ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-700 hover:bg-gray-300"}`}>
                                    {metricLabels[m]}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="min-h-[600px] flex items-center justify-center">
                    {loading && <p>Loading...</p>}
                    {!loading && !plotConfig && sampleId && <p className="text-gray-400">No heatmap data available for this sample.</p>}
                    {!loading && plotConfig && (
                        <DynamicPlot
                            data={plotConfig.data}
                            layout={{ ...plotConfig.layout, autosize: true, width: undefined }}
                            useResizeHandler={true}
                            style={{ width: "100%", height: "100%" }}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
