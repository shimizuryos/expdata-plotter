"use client";
import { useState, useEffect, useRef } from "react";
import DynamicPlot from "@/components/DynamicPlot";

const API = "http://localhost:8000/api/data";

interface Props { samples: any[] }

export default function IVSingleTab({ samples }: Props) {
    const [sampleId, setSampleId] = useState("");
    const [devices, setDevices] = useState<any[]>([]);
    const [selectedDeviceId, setSelectedDeviceId] = useState("");
    const [measurements, setMeasurements] = useState<any[]>([]);
    const [checkedMeas, setCheckedMeas] = useState<Set<string>>(new Set());
    const [plotMode, setPlotMode] = useState<"logR" | "logRA">("logR");
    const [rPara, setRPara] = useState<number>(0); // Added r_parasitic state
    const [ivPlot, setIvPlot] = useState<any>(null);
    const [rvPlot, setRvPlot] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const rawRvDataRef = useRef<{ plot: any; area: number } | null>(null);

    useEffect(() => {
        if (!sampleId) return;
        // Load devices AND sample info for r_parasitic
        Promise.all([
            fetch(`${API}/samples/${sampleId}/device-groups`).then(r => r.json()),
            fetch(`${API}/samples/${sampleId}`).then(r => r.json())
        ]).then(([groups, sample]) => {
            const devs: any[] = [];
            for (const g of groups) for (const d of g.devices) devs.push({ ...d, coord: g.coord });
            setDevices(devs);
            if (sample.r_parasitic != null) setRPara(sample.r_parasitic);
            else setRPara(0);
        });
        setSelectedDeviceId(""); setMeasurements([]); setCheckedMeas(new Set());
    }, [sampleId]);

    useEffect(() => {
        if (!selectedDeviceId || !sampleId) return;
        fetch(`${API}/samples/${sampleId}/devices/${selectedDeviceId}/measurements`)
            .then(r => r.json()).then(ms => {
                const ivMs = ms.filter((m: any) => m.measurement_type === "IV");
                setMeasurements(ivMs);
                setCheckedMeas(new Set());
            });
    }, [selectedDeviceId]);

    const toggleMeas = (id: string) => {
        setCheckedMeas(prev => {
            const next = new Set(prev);
            next.has(id) ? next.delete(id) : next.add(id);
            return next;
        });
    };

    const loadPlots = async () => {
        const selected = measurements.filter(m => checkedMeas.has(m.id));
        if (selected.length === 0) return;
        setLoading(true);

        const device = devices.find(d => d.device_id === selectedDeviceId);
        const area = device?.area_um2 || 1;

        if (selected.length === 1) {
            const m = selected[0];
            const res = await fetch(`${API}/iv/load`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file_ref: m.file_ref, area_um2: area, label: selectedDeviceId, r_p: rPara })
            });
            const data = await res.json();
            setIvPlot(data.iv_plot);
            rawRvDataRef.current = { plot: data.log_r_v_plot, area };
            applyRvMode(data.log_r_v_plot, area);
        } else {
            const entries = selected.map(m => ({
                file_ref: m.file_ref, area_um2: area,
                label: `${selectedDeviceId} (${m.measured_at?.substring(0, 10) || m.id.substring(0, 8)})`,
                r_p: rPara
            }));
            const res = await fetch(`${API}/iv/load-multi`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ entries })
            });
            const data = await res.json();
            setIvPlot(data.iv_plot);
            rawRvDataRef.current = { plot: data.log_r_v_plot, area };
            applyRvMode(data.log_r_v_plot, area);
        }
        setLoading(false);
    };

    const applyRvMode = (basePlot: any, area: number) => {
        if (plotMode === "logRA" && basePlot) {
            const modified = { ...basePlot };
            modified.data = basePlot.data.map((trace: any) => ({
                ...trace,
                y: trace.y.map((r: number) => r * area),
            }));
            modified.layout = { ...basePlot.layout, yaxis: { ...basePlot.layout.yaxis, title: { text: "RA (Ω·μm²)" } } };
            modified.layout.title = "Log RA vs V";
            setRvPlot(modified);
        } else {
            setRvPlot(basePlot);
        }
    };

    useEffect(() => {
        if (rawRvDataRef.current) {
            applyRvMode(rawRvDataRef.current.plot, rawRvDataRef.current.area);
        }
    }, [plotMode]);

    return (
        <div className="space-y-4">
            {/* Selectors */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-semibold mb-1">Sample</label>
                    <select className="w-full p-2 border rounded" value={sampleId} onChange={e => setSampleId(e.target.value)}>
                        <option value="">-- Select --</option>
                        {samples.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                    </select>
                </div>
                <div>
                    <label className="block text-sm font-semibold mb-1">Device</label>
                    <select className="w-full p-2 border rounded" value={selectedDeviceId} onChange={e => setSelectedDeviceId(e.target.value)}>
                        <option value="">-- Select --</option>
                        {devices.map(d => <option key={d.device_id} value={d.device_id}>{d.device_id} ({d.area_um2} μm²)</option>)}
                    </select>
                </div>
            </div>

            {/* Measurement Checkboxes */}
            {measurements.length > 0 && (
                <div className="bg-gray-50 p-3 rounded border">
                    <h3 className="text-xs font-bold text-gray-500 mb-2">IV MEASUREMENTS</h3>
                    {measurements.map(m => (
                        <label key={m.id} className="flex items-center gap-2 text-sm py-1 cursor-pointer hover:bg-gray-100 px-2 rounded">
                            <input type="checkbox" checked={checkedMeas.has(m.id)} onChange={() => toggleMeas(m.id)} />
                            <span>{m.file_ref?.split('/').pop()} ({m.measured_at?.substring(0, 10) || "N/A"})</span>
                        </label>
                    ))}
                    <button onClick={loadPlots} disabled={checkedMeas.size === 0 || loading}
                        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
                        {loading ? "Loading..." : "Load & Plot"}
                    </button>
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
                            <button onClick={() => setPlotMode(plotMode === "logR" ? "logRA" : "logR")}
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
