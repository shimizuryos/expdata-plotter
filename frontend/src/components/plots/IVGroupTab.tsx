"use client";
import { useState, useEffect, useRef } from "react";
import DynamicPlot from "@/components/DynamicPlot";

const API = "http://localhost:8000/api/data";

interface Props { samples: any[] }

export default function IVGroupTab({ samples }: Props) {
    const [sampleId, setSampleId] = useState("");
    const [groups, setGroups] = useState<any[]>([]);
    const [selectedCoord, setSelectedCoord] = useState("");
    const [devicesInGroup, setDevicesInGroup] = useState<any[]>([]);
    const [checkedDevices, setCheckedDevices] = useState<Set<string>>(new Set());
    const [plotMode, setPlotMode] = useState<"logR" | "logRA">("logRA");
    const [ivPlot, setIvPlot] = useState<any>(null);
    const [rvPlot, setRvPlot] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    // Fitting state
    const [rPara, setRPara] = useState<number>(0);
    const [fittingRms, setFittingRms] = useState<number | null>(null); // fittingRms is effectively unused / null with analytic fit but kept for type safety or future use
    const [isFitting, setIsFitting] = useState(false);
    const fittingAbortRef = useRef(false);

    useEffect(() => {
        if (!sampleId) return;
        fetch(`${API}/samples/${sampleId}/device-groups`).then(r => r.json()).then(gs => {
            setGroups(gs);
            // Load sample r_parasitic
            fetch(`${API}/samples/${sampleId}`).then(r => r.json()).then(s => {
                if (s.r_parasitic != null) setRPara(s.r_parasitic);
            });
        });
        setSelectedCoord(""); setDevicesInGroup([]); setCheckedDevices(new Set());
    }, [sampleId]);

    useEffect(() => {
        if (!selectedCoord) return;
        const g = groups.find((g: any) => JSON.stringify(g.coord) === selectedCoord);
        if (g) {
            setDevicesInGroup(g.devices);
            setCheckedDevices(new Set(g.devices.map((d: any) => d.device_id)));
        }
    }, [selectedCoord]);

    const toggleDevice = (id: string) => {
        setCheckedDevices(prev => {
            const n = new Set(prev);
            n.has(id) ? n.delete(id) : n.add(id);
            return n;
        });
    };

    const buildEntries = async () => {
        const entries: any[] = [];
        for (const dev of devicesInGroup) {
            if (!checkedDevices.has(dev.device_id)) continue;
            const res = await fetch(`${API}/samples/${sampleId}/devices/${dev.device_id}/measurements`);
            const ms = await res.json();
            const ivMs = ms.filter((m: any) => m.measurement_type === "IV");
            const defaultId = dev.default_measurements?.["IV"];
            const meas = defaultId ? ivMs.find((m: any) => m.id === defaultId) : ivMs[0];
            if (meas) {
                entries.push({ file_ref: meas.file_ref, area_um2: dev.area_um2, label: dev.device_id });
            }
        }
        return entries;
    };

    const loadPlots = async (rsCorrection: number = 0) => {
        setLoading(true);
        const entries = await buildEntries();
        if (entries.length === 0) { setLoading(false); return; }

        const entriesWithRs = entries.map(e => ({ ...e, r_p: rPara }));

        const res = await fetch(`${API}/iv/load-multi`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entries: entriesWithRs })
        });
        const data = await res.json();
        setIvPlot(data.iv_plot);

        // For group plot, default to RA mode
        if (plotMode === "logRA") {
            const modified = { ...data.log_r_v_plot };
            modified.data = data.log_r_v_plot.data.map((trace: any, i: number) => ({
                ...trace,
                y: trace.y.map((r: number) => r * (entriesWithRs[i]?.area_um2 || 1)),
            }));
            modified.layout = { ...data.log_r_v_plot.layout, yaxis: { ...data.log_r_v_plot.layout.yaxis, title: { text: "RA (Ω·μm²)" } }, title: "Log RA vs V" };
            setRvPlot(modified);
        } else {
            setRvPlot(data.log_r_v_plot);
        }
        setLoading(false);
    };

    const handleFit = async () => {
        const entries = await buildEntries();
        if (entries.length < 2) { alert("Need at least 2 devices for fitting"); return; }

        setIsFitting(true);
        fittingAbortRef.current = false;

        try {
            const res = await fetch(`${API}/iv/fit-parasitic`, {
                method: "POST", headers: { "Content-Type": "application/json" }, // Removed initial_ra_para from body as it is not needed/ used cleanly for analytic
                body: JSON.stringify({ entries, mode: "full" })
            });
            if (fittingAbortRef.current) return;

            const data = await res.json();
            setRPara(data.r_para);
            setFittingRms(null); // No RMS in analytic solution

            // Apply fitted plots
            if (data.plots) {
                setIvPlot(data.plots.iv_plot);
                if (plotMode === "logRA") {
                    const modified = { ...data.plots.log_r_v_plot };
                    modified.data = data.plots.log_r_v_plot.data.map((trace: any, i: number) => ({
                        ...trace,
                        y: trace.y.map((r: number) => r * (entries[i]?.area_um2 || 1)),
                    }));
                    modified.layout = { ...data.plots.log_r_v_plot.layout, yaxis: { ...data.plots.log_r_v_plot.layout.yaxis, title: { text: "RA (Ω·μm²)" } }, title: "Log RA vs V (Fitted)" };
                    setRvPlot(modified);
                } else {
                    setRvPlot(data.plots.log_r_v_plot);
                }
            }
        } finally {
            setIsFitting(false);
        }
    };

    const handleStopFitting = () => {
        fittingAbortRef.current = true;
        setIsFitting(false);
    };

    const handleSaveRaPara = async () => {
        await fetch(`${API}/samples/${sampleId}/r-parasitic`, {
            method: "PUT", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ r_parasitic: rPara })
        });
        alert("R parasitic saved!");
    };

    const handleRaParaChange = (val: number) => {
        setRPara(val);
    };

    const handleRaParaApply = () => {
        loadPlots();
    };

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-semibold mb-1">Sample</label>
                    <select className="w-full p-2 border rounded" value={sampleId} onChange={e => setSampleId(e.target.value)}>
                        <option value="">-- Select --</option>
                        {samples.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-semibold mb-1">Device Group</label>
                    <select className="w-full p-2 border rounded" value={selectedCoord} onChange={e => setSelectedCoord(e.target.value)}>
                        <option value="">-- Select --</option>
                        {groups.map(g => (
                            <option key={JSON.stringify(g.coord)} value={JSON.stringify(g.coord)}>
                                Coord: ({g.coord[0]}, {g.coord[1]}) - {g.devices.length} devices
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {devicesInGroup.length > 0 && (
                <div className="bg-gray-50 p-3 rounded border">
                    <h3 className="text-xs font-bold text-gray-500 mb-2">DEVICES IN GROUP (uncheck to exclude)</h3>
                    <div className="flex flex-wrap gap-2">
                        {devicesInGroup.map(d => (
                            <label key={d.device_id} className="flex items-center gap-1 text-sm bg-white px-2 py-1 rounded border cursor-pointer hover:bg-blue-50">
                                <input type="checkbox" checked={checkedDevices.has(d.device_id)} onChange={() => toggleDevice(d.device_id)} />
                                {d.device_id} ({d.area_um2} μm²)
                            </label>
                        ))}
                    </div>
                    <button onClick={() => loadPlots()} disabled={checkedDevices.size === 0 || loading}
                        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
                        {loading ? "Loading..." : "Load & Plot Group"}
                    </button>
                </div>
            )}

            {/* Fitting Controls */}
            {ivPlot && (
                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded space-y-3">
                    <h3 className="font-bold text-sm">Parasitic Resistance Fitting (Analytic)</h3>
                    <div className="flex items-center gap-3">
                        <label className="text-sm font-medium whitespace-nowrap">R_para (Ω):</label>
                        <input type="number" step="0.1" value={rPara}
                            onChange={e => handleRaParaChange(parseFloat(e.target.value) || 0)}
                            className="w-32 p-2 border rounded font-mono text-sm" />
                        <button onClick={handleRaParaApply}
                            className="px-3 py-2 bg-gray-600 text-white rounded text-sm hover:bg-gray-700">Apply</button>
                        <button onClick={handleFit} disabled={isFitting}
                            className="px-3 py-2 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50">
                            {isFitting ? "Fitting..." : "Auto Calc R_para"}
                        </button>
                        {isFitting && (
                            <button onClick={handleStopFitting}
                                className="px-3 py-2 bg-red-600 text-white rounded text-sm hover:bg-red-700">Stop</button>
                        )}
                        <button onClick={handleSaveRaPara}
                            className="px-3 py-2 bg-purple-600 text-white rounded text-sm hover:bg-purple-700 ml-auto">Save to Sample</button>
                    </div>
                </div>
            )}

            {/* Plots */}
            {ivPlot && (
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white p-2 rounded border">
                        <DynamicPlot data={ivPlot.data} layout={{ ...ivPlot.layout, autosize: true, width: undefined, height: 500 }}
                            useResizeHandler={true} style={{ width: "100%", height: "100%" }} />
                    </div>
                    <div className="bg-white p-2 rounded border">
                        <div className="flex justify-end mb-1">
                            <button onClick={() => { setPlotMode(plotMode === "logR" ? "logRA" : "logR"); loadPlots(); }}
                                className="text-xs px-3 py-1 bg-gray-200 rounded hover:bg-gray-300">
                                {plotMode === "logR" ? "Switch to log RA-V" : "Switch to log R-V"}
                            </button>
                        </div>
                        {rvPlot && <DynamicPlot data={rvPlot.data} layout={{ ...rvPlot.layout, autosize: true, width: undefined, height: 480 }}
                            useResizeHandler={true} style={{ width: "100%", height: "100%" }} />}
                    </div>
                </div>
            )}
        </div>
    );
}
