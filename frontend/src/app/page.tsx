import Link from "next/link";

export default function Home() {
    return (
        <main className="flex min-h-screen flex-col items-center justify-center p-24">
            <h1 className="text-4xl font-bold mb-8">Research Data Plotter</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
                <Link
                    href="/upload"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Upload Data</h2>
                    <p className="text-gray-600">Upload new YAML/CSV data files.</p>
                </Link>

                <Link
                    href="/plots/ps-ra"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Ps-RA Plot</h2>
                    <p className="text-gray-600">Interactive Ps vs RA visualization.</p>
                </Link>

                <Link
                    href="/plots/iv"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">IV Characteristics</h2>
                    <p className="text-gray-600">IV curves visualization.</p>
                </Link>

                <Link
                    href="/plots/hanle"
                    className="p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200"
                >
                    <h2 className="text-2xl font-semibold mb-2">Hanle Effect</h2>
                    <p className="text-gray-600">Hanle signal visualization.</p>
                </Link>
            </div>
        </main>
    );
}
