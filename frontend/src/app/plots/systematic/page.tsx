"use client";

import Link from "next/link";

export default function SystematicPlotsPage() {
    return (
        <main className="flex min-h-screen flex-col items-center p-24">
            <h1 className="text-4xl font-bold mb-8">Systematic Plots</h1>
            <p className="mb-12 text-gray-600">Select a plot type to visualize specific data relationships.</p>

            <div className="grid grid-cols-1 gap-6 w-full max-w-2xl">
                <Link
                    href="/plots/ps-ra"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Ps vs RA Plot</h2>
                    <p className="text-gray-600">Visualize Point data (Ps, RA, RMS) from summary YAML.</p>
                </Link>

                <Link
                    href="/plots/log-ra-v"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Log RA vs V Plot</h2>
                    <p className="text-gray-600">Visualize Log(RA) vs Voltage curves for grouped experimental data.</p>
                </Link>

                <Link
                    href="/plots/systematic/heatmap"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Heatmap</h2>
                    <p className="text-gray-600">Visualize Ps, RA, RMS values on an (x, y) coordinate grid.</p>
                </Link>
            </div>

            <Link href="/" className="mt-12 px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                Back to Home
            </Link>
        </main>
    );
}
