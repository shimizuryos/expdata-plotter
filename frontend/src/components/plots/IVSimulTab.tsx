"use client";
import { useState, useEffect } from "react";
import DynamicPlot from "@/components/DynamicPlot";

const API = "http://localhost:8000/api/data";

interface Props { samples: any[] }

export default function IVSimulTab({ samples }: Props) {
    const [sampleId, setSampleId] = useState("");
    const [devices, setDevices] = useState<any[]>([]);
    const [checkedDevices, setCheckedDevices] = useState<Set<string>>(new Set());
    const [plotMode, setPlotMode] = useState<"logR" | "logRA">("logR");
    const [ivPlot, setIvPlot] = useState<any>(null);
    const [rvPlot, setRvPlot] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [rPara, setRPara] = useState<number>(0);

    useEffect(() => {
        if (!sampleId) return;
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
        setCheckedDevices(new Set());
    }, [sampleId]);

    const toggleDevice = (id: string) => {
        setCheckedDevices(prev => {
            const n = new Set(prev);
            n.has(id) ? n.delete(id) : n.add(id);
            return n;
        });
    };

    const loadPlots = async () => {
        setLoading(true);
        // For each checked device, find its default IV measurement or latest IV
        const entries: any[] = [];
        for (const devId of checkedDevices) {
            const dev = devices.find(d => d.device_id === devId);
            if (!dev) continue;
            // Get measurements
            const res = await fetch(`${API}/samples/${sampleId}/devices/${devId}/measurements`);
            const ms = await res.json();
            const ivMs = ms.filter((m: any) => m.measurement_type === "IV");
            // Use default or latest
            const defaultId = dev.default_measurements?.["IV"];
            const meas = defaultId ? ivMs.find((m: any) => m.id === defaultId) : ivMs[0];
            if (meas) {
                entries.push({
                    file_ref: meas.file_ref,
                    area_um2: dev.area_um2,
                    label: devId,
                    r_p: rPara,
                });
            }
        }

        if (entries.length === 0) { setLoading(false); return; }

        const res = await fetch(`${API}/iv/load-multi`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entries })
        });
        const data = await res.json();
        setIvPlot(data.iv_plot);
        transformRvPlot(data.log_r_v_plot, entries);
        setLoading(false);
    };

    const transformRvPlot = (basePlot: any, entries: any[]) => {
        if (plotMode === "logRA" && basePlot) {
            const modified = { ...basePlot };
            modified.data = basePlot.data.map((trace: any, i: number) => ({
                ...trace,
                y: trace.y.map((r: number) => r * (entries[i]?.area_um2 || 1)),
            }));
            modified.layout = { ...basePlot.layout, yaxis: { ...basePlot.layout.yaxis, title: { text: "RA (Ω·μm²)" } }, title: "Log RA vs V" };
            setRvPlot(modified);
        } else {
            setRvPlot(basePlot);
        }
    };

    return (
        <div className="space-y-4">
            <div>
                <label className="block text-sm font-semibold mb-1">Sample</label>
                <select className="w-full p-2 border rounded" value={sampleId} onChange={e => setSampleId(e.target.value)}>
                    <option value="">-- Select --</option>
                    {samples.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
                </select>
            </div>

            {devices.length > 0 && (
                <div className="bg-gray-50 p-3 rounded border">
                    <h3 className="text-xs font-bold text-gray-500 mb-2">SELECT DEVICES (check multiple)</h3>
                    <div className="grid grid-cols-3 gap-1 max-h-48 overflow-y-auto">
                        {devices.map(d => (
                            <label key={d.device_id} className="flex items-center gap-2 text-sm py-1 cursor-pointer hover:bg-gray-100 px-2 rounded">
                                <input type="checkbox" checked={checkedDevices.has(d.device_id)} onChange={() => toggleDevice(d.device_id)} />
                                <span>{d.device_id} ({d.area_um2})</span>
                            </label>
                        ))}
                    </div>
                    <button onClick={loadPlots} disabled={checkedDevices.size === 0 || loading}
                        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm">
                        {loading ? "Loading..." : `Plot ${checkedDevices.size} Devices`}
                    </button>
                </div>
            )}

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
