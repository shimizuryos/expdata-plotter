"use client";

import { useState } from "react";
import Link from "next/link";

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setMessage(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("http://localhost:8000/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                throw new Error("Upload failed");
            }

            const data = await res.json();
            setMessage(`Success: ${data.info}`);
        } catch (err: any) {
            setMessage(`Error: ${err.message}`);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="flex flex-col items-center min-h-screen p-8">
            <div className="w-full max-w-2xl flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold">Upload Data</h1>
                <Link href="/" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
                    Back to Home
                </Link>
            </div>

            <div className="w-full max-w-xl bg-white p-8 rounded-xl shadow-lg border border-gray-200">
                <div className="mb-6">
                    <label className="block text-gray-700 text-sm font-bold mb-2">
                        Select File
                    </label>
                    <input
                        type="file"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
                    />
                </div>

                <button
                    onClick={handleUpload}
                    disabled={!file || uploading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
                >
                    {uploading ? "Uploading..." : "Upload"}
                </button>

                {message && (
                    <div className={`mt-4 p-4 rounded ${message.startsWith("Error") ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}>
                        {message}
                    </div>
                )}
            </div>
        </div>
    );
}
