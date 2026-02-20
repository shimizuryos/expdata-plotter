import sys

filepath = "/Users/shimizuryousuke/dev/expdata-plotter/plot-src/expdata-plotter/frontend/src/app/plots/ps-ra/page.tsx"

new_code = """\"\"\"use client\"\"\";
import { useEffect, useState, useRef, useCallback } from "react";
import DynamicPlot from "@/components/DynamicPlot";
import Link from "next/link";

const API = "http://localhost:8000/api";
const STORAGE_KEY = "ps-ra-fit-params-v2";

interface SeriesMeta {
    label: string;
    color: string;
    r_p: number;
    n_points: number;
}

export default function PsRaPlotPage() {
    const [plotData, setPlotData] = useState<any>(null);
    const [seriesList, setSeriesList] = useState<SeriesMeta[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fitting state
    const [selectedSeries, setSelectedSeries] = useState("");
    
    const [fixPsA, setFixPsA] = useState(true);
    const [fixPsB, setFixPsB] = useState(true);
    const [fixLamA, setFixLamA] = useState(false);
    const [fixLamB, setFixLamB] = useState(false);
    const [fixC, setFixC] = useState(false);
    const [fixDB, setFixDB] = useState(false);

    const [valPsA, setValPsA] = useState("0.5");
    const [valPsB, setValPsB] = useState("0.3");
    const [valLamA, setValLamA] = useState("1e-9");
    const [valLamB, setValLamB] = useState("5e-9");
    const [valC, setValC] = useState("1.0");
    const [valDB, setValDB] = useState("1.0");
    const [valVB, setValVB] = useState("0.1");

    const [weightRatio, setWeightRatio] = useState("1.0");

    // Fit results
    const [fitting, setFitting] = useState(false);
    const [previewing, setPreviewing] = useState(false);
    const [fitProgress, setFitProgress] = useState<any>(null);
    const [fitResult, setFitResult] = useState<any>(null);
    const pollRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        fetch(`${API}/plots/ps-ra`)
            .then(async (res) => {
                if (!res.ok) throw new Error(`Failed: ${res.status}`);
                return res.json();
            })
            .then((data) => {
                setPlotData(data.plot);
                setSeriesList(data.series || []);
                // Restore saved params
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) {
                    try {
                        const p = JSON.parse(saved);
                        if (p.selectedSeries) setSelectedSeries(p.selectedSeries);
                        if (p.valPsA) setValPsA(p.valPsA);
                        if (p.valPsB) setValPsB(p.valPsB);
                        if (p.valLamA) setValLamA(p.valLamA);
                        if (p.valLamB) setValLamB(p.valLamB);
                        if (p.valC) setValC(p.valC);
                        if (p.valDB) setValDB(p.valDB);
                        if (p.valVB) setValVB(p.valVB);
                        if (p.weightRatio) setWeightRatio(p.weightRatio);
                        
                        if (p.fixPsA !== undefined) setFixPsA(p.fixPsA);
                        if (p.fixPsB !== undefined) setFixPsB(p.fixPsB);
                        if (p.fixLamA !== undefined) setFixLamA(p.fixLamA);
                        if (p.fixLamB !== undefined) setFixLamB(p.fixLamB);
                        if (p.fixC !== undefined) setFixC(p.fixC);
                        if (p.fixDB !== undefined) setFixDB(p.fixDB);
                    } catch { /* ignore bad data */ }
                } else if (data.series?.length > 0) {
                    setSelectedSeries(data.series[0].label);
                }
                setLoading(false);
            })
            .catch((err) => { setError(err.message); setLoading(false); });
    }, []);

    const saveParams = () => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            selectedSeries, 
            fixPsA, fixPsB, fixLamA, fixLamB, fixC, fixDB,
            valPsA, valPsB, valLamA, valLamB, valC, valDB, valVB,
            weightRatio
        }));
    };

    const handlePreview = useCallback(async () => {
        saveParams();
        setPreviewing(true);
        try {
            const body = {
                series_label: selectedSeries,
                init_ps_a: parseFloat(valPsA),
                init_ps_b: parseFloat(valPsB),
                init_lam_a: parseFloat(valLamA) || 1e-9,
                init_lam_b: parseFloat(valLamB) || 5e-9,
                init_c: parseFloat(valC) || 1.0,
                init_d_b: parseFloat(valDB) || 1.0,
                V_B: parseFloat(valVB) || 0.1,
            };
            const res = await fetch(`${API}/plots/ps-ra/preview`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            setFitResult(data);
        } catch (err: any) {
            setError(err.message);
        }
        setPreviewing(false);
    }, [selectedSeries, valPsA, valPsB, valLamA, valLamB, valC, valDB, valVB]);

    const startFit = useCallback(async () => {
        saveParams();
        setFitting(true);
        setFitResult(null);
        setFitProgress(null);

        const body: any = {
            series_label: selectedSeries,
            fix_ps_a: fixPsA, fix_ps_b: fixPsB,
            fix_lam_a: fixLamA, fix_lam_b: fixLamB,
            fix_c: fixC, fix_d_b: fixDB,
            init_ps_a: parseFloat(valPsA),
            init_ps_b: parseFloat(valPsB),
            init_lam_a: parseFloat(valLamA) || 1e-9,
            init_lam_b: parseFloat(valLamB) || 5e-9,
            init_c: parseFloat(valC) || 1.0,
            init_d_b: parseFloat(valDB) || 1.0,
            V_B: parseFloat(valVB) || 0.1,
            weight_ratio: parseFloat(weightRatio),
        };

        try {
            const res = await fetch(`${API}/plots/ps-ra/fit`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const { job_id } = await res.json();

            pollRef.current = setInterval(async () => {
                try {
                    const pr = await fetch(`${API}/plots/ps-ra/fit/${job_id}`);
                    const data = await pr.json();
                    if (data.status === "done") {
                        clearInterval(pollRef.current!);
                        pollRef.current = null;
                        setFitResult(data);
                        setFitProgress(null);
                        setFitting(false);
                    } else if (data.status === "error") {
                        clearInterval(pollRef.current!);
                        pollRef.current = null;
                        setFitProgress({ ...data });
                        setFitting(false);
                    } else {
                        setFitProgress(data);
                    }
                } catch { /* keep polling */ }
            }, 500);
        } catch (err: any) {
            setError(err.message);
            setFitting(false);
        }
    }, [selectedSeries, fixPsA, fixPsB, fixLamA, fixLamB, fixC, fixDB, valPsA, valPsB, valLamA, valLamB, valC, valDB, valVB, weightRatio]);

    useEffect(() => {
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, []);

    const inputCls = "w-full px-2 py-1.5 border rounded text-sm bg-white focus:ring-1 focus:ring-blue-400 outline-none";
    const labelCls = "block text-xs font-medium text-gray-600 mb-0";

    return (
        <div className="min-h-screen bg-gray-50 p-6">
            <div className="max-w-[1400px] mx-auto">
                {/* Header */}
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-2xl font-bold text-gray-800">Ps vs RA — Fitting</h1>
                    <Link href="/plots/systematic" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm">
                        Back to Systematic Plots
                    </Link>
                </div>

                {loading && <p className="text-gray-500">Loading data...</p>}
                {error && <p className="text-red-500">Error: {error}</p>}

                {!loading && !error && plotData && (
                    <div className="grid grid-cols-[320px_1fr] gap-6">
                        {/* ── Left Panel: Controls ── */}
                        <div className="space-y-4">
                            {/* Series Selection */}
                            <div className="bg-white rounded-lg border p-4 shadow-sm">
                                <h3 className="text-sm font-bold text-gray-700 mb-3">Target Series</h3>
                                <select className={inputCls} value={selectedSeries}
                                    onChange={e => setSelectedSeries(e.target.value)}>
                                    {seriesList.map(s => (
                                        <option key={s.label} value={s.label}>
                                            {s.label} ({s.n_points} pts, R_p={s.r_p} Ω)
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Weight */}
                            <div className="bg-white rounded-lg border p-4 shadow-sm">
                                <h3 className="text-sm font-bold text-gray-700 mb-3">Cost Weighting</h3>
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <label className={labelCls}>σ_lnR / σ_P</label>
                                        <span className="font-mono text-xs text-blue-600">{weightRatio}</span>
                                    </div>
                                    <input type="range" min="0.1" max="10" step="0.1"
                                        className="w-full" value={weightRatio}
                                        onChange={e => setWeightRatio(e.target.value)} />
                                    <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                                        <span>← RA fit focus</span>
                                        <span>Ps fit focus →</span>
                                    </div>
                                </div>
                            </div>

                            {/* Parameters */}
                            <div className="bg-white rounded-lg border p-4 shadow-sm">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-sm font-bold text-gray-700">Parameters</h3>
                                    <span className="text-[10px] text-gray-400">Check to Fix</span>
                                </div>
                                
                                <div className="space-y-3">
                                    <div className="grid grid-cols-[auto_60px_1fr] gap-3 items-center">
                                        <div className="w-4"></div>
                                        <label className={labelCls}>D_A</label>
                                        <input className={inputCls} value="1.0" disabled title="D_A is uniformly fixed" />
                                    </div>

                                    <div className="grid grid-cols-[auto_60px_1fr] gap-3 items-center">
                                        <div className="w-4"></div>
                                        <label className={labelCls}>V_B</label>
                                        <input className={inputCls} value={valVB} onChange={e => setValVB(e.target.value)} title="Bias Voltage" />
                                    </div>
                                    
                                    <hr className="my-3 border-gray-100" />

                                    {[
                                        { id: 'ps_a', label: 'P_S,A', fix: fixPsA, setFix: setFixPsA, val: valPsA, setVal: setValPsA },
                                        { id: 'ps_b', label: 'P_S,B', fix: fixPsB, setFix: setFixPsB, val: valPsB, setVal: setValPsB },
                                        { id: 'lam_a', label: 'λ_A (m)', fix: fixLamA, setFix: setFixLamA, val: valLamA, setVal: setValLamA },
                                        { id: 'lam_b', label: 'λ_B (m)', fix: fixLamB, setFix: setFixLamB, val: valLamB, setVal: setValLamB },
                                        { id: 'c', label: 'C', fix: fixC, setFix: setFixC, val: valC, setVal: setValC },
                                        { id: 'd_b', label: 'D_B', fix: fixDB, setFix: setFixDB, val: valDB, setVal: setValDB },
                                    ].map(p => (
                                        <div key={p.id} className="grid grid-cols-[auto_60px_1fr] gap-3 items-center">
                                            <input type="checkbox" checked={p.fix} onChange={e => p.setFix(e.target.checked)} className="w-4 h-4 cursor-pointer" title={`Fix ${p.label}`} />
                                            <label className={labelCls}>{p.label}</label>
                                            <input className={inputCls} value={p.val} onChange={e => p.setVal(e.target.value)} />
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Action Buttons */}
                            <div className="flex gap-2">
                                <button onClick={handlePreview} disabled={fitting || previewing || !selectedSeries}
                                    className="flex-1 py-2.5 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 font-semibold text-sm shadow">
                                    {previewing ? "Loading..." : "Preview"}
                                </button>
                                <button onClick={startFit} disabled={fitting || !selectedSeries}
                                    className="flex-1 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-semibold text-sm shadow">
                                    {fitting ? "Fitting..." : "Start Fit"}
                                </button>
                            </div>

                            {/* Progress */}
                            {fitting && fitProgress && (
                                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm">
                                    <p className="font-mono text-xs">Iteration: {fitProgress.iteration} | Cost: {fitProgress.cost?.toFixed(4)}</p>
                                    {fitProgress.params && (
                                        <div className="mt-1 text-xs text-gray-600 space-y-0.5">
                                            <div>D_B = {fitProgress.params.D_B?.toExponential(2)}</div>
                                            <div>C = {fitProgress.params.C?.toExponential(2)}</div>
                                            <div>λ_A = {fitProgress.params.lambda_A?.toExponential(2)} | λ_B = {fitProgress.params.lambda_B?.toExponential(2)}</div>
                                            <div>P_S,A = {fitProgress.params.P_S_A?.toFixed(3)} | P_S,B = {fitProgress.params.P_S_B?.toFixed(3)}</div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Results */}
                            {fitResult && (fitResult.status === "done" || fitResult.status === "preview") && (
                                <div className={`${fitResult.status === "preview" ? "bg-blue-50 border-blue-200" : "bg-green-50 border-green-200"} border rounded-lg p-3 text-sm`}>
                                    <h4 className={`font-bold mb-2 pb-1 border-b ${fitResult.status === "preview" ? "text-blue-700 border-blue-200" : "text-green-700 border-green-200"}`}>
                                        {fitResult.status === "preview" ? "Preview Params" : "Fit Results"}
                                    </h4>
                                    <table className="text-xs w-full">
                                        <tbody>
                                            <tr><td className="font-mono pr-2 text-gray-500">D_A</td><td className="font-mono">{(1.0).toExponential(4)}</td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">D_B</td><td className="font-mono">{fitResult.params.D_B?.toExponential(4)}</td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">C</td><td className="font-mono">{fitResult.params.C?.toExponential(4)}</td></tr>
                                            <tr><td className="col-span-2 pt-1"></td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">λ_A</td><td className="font-mono">{fitResult.params.lambda_A?.toExponential(4)} m</td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">λ_B</td><td className="font-mono">{fitResult.params.lambda_B?.toExponential(4)} m</td></tr>
                                            <tr><td className="col-span-2 pt-1"></td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">P_S,A</td><td className="font-mono">{fitResult.params.P_S_A?.toFixed(4)}</td></tr>
                                            <tr><td className="font-mono pr-2 text-gray-500">P_S,B</td><td className="font-mono">{fitResult.params.P_S_B?.toFixed(4)}</td></tr>
                                        </tbody>
                                    </table>
                                    {fitResult.info && (<>
                                        <hr className={`my-2 ${fitResult.status === "preview" ? "border-blue-200" : "border-green-200"}`} />
                                        <p className="text-[10px] text-gray-500 font-mono">
                                            Cost: {fitResult.info?.cost?.toFixed(6)}<br />
                                            Residual: {fitResult.info?.residual_norm?.toFixed(4)}<br />
                                            Evals: {fitResult.info?.nfev}
                                        </p>
                                    </>)}
                                </div>
                            )}

                            {fitResult && fitResult.status === "error" && (
                                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700 break-words">
                                    <span className="font-bold">Error:</span> {fitResult.error}
                                </div>
                            )}
                        </div>

                        {/* ── Right Panel: Plots ── */}
                        <div className="space-y-4">
                            {/* Original Ps vs RA plot */}
                            {!fitResult?.plots && (
                                <div className="bg-white rounded-lg border p-3 shadow-sm h-[500px]">
                                    <DynamicPlot data={plotData.data}
                                        layout={{ ...plotData.layout, autosize: true, width: undefined, height: 500 }}
                                        useResizeHandler={true} style={{ width: "100%", height: "100%" }} />
                                </div>
                            )}

                            {/* Fit result plots */}
                            {fitResult?.plots && (
                                <>
                                    <div className="bg-white rounded-lg border p-3 shadow-sm">
                                        <DynamicPlot data={fitResult.plots.ps_ra.data}
                                            layout={{ ...fitResult.plots.ps_ra.layout, autosize: true, width: undefined, height: 450 }}
                                            useResizeHandler={true} style={{ width: "100%", height: "100%" }} />
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-white rounded-lg border p-3 shadow-sm">
                                            <DynamicPlot data={fitResult.plots.ps_tox.data}
                                                layout={{ ...fitResult.plots.ps_tox.layout, autosize: true, width: undefined, height: 380 }}
                                                useResizeHandler={true} style={{ width: "100%", height: "100%" }} />
                                        </div>
                                        <div className="bg-white rounded-lg border p-3 shadow-sm">
                                            <DynamicPlot data={fitResult.plots.ra_tox.data}
                                                layout={{ ...fitResult.plots.ra_tox.layout, autosize: true, width: undefined, height: 380 }}
                                                useResizeHandler={true} style={{ width: "100%", height: "100%" }} />
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
"""

# NextJS 'use client' must not be quoted incorrectly. Oh wait I did \"\"\"use client\"\"\";
# Let's fix it by replacing the first line properly.
new_code = new_code.replace('\"\"\"use client\"\"\";', '\"use client\";')

with open(filepath, "w") as f:
    f.write(new_code)
