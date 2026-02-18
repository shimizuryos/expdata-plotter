"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { SampleSection } from "@/components/upload/SampleSection";
import { DeviceGroupSection } from "@/components/upload/DeviceGroupSection";
import { DeviceSection } from "@/components/upload/DeviceSection";
import { MeasurementSection } from "@/components/upload/MeasurementSection";

export default function UploadPage() {
    // --- State ---
    const [samples, setSamples] = useState<any[]>([]);
    const [selectedSampleId, setSelectedSampleId] = useState<string>("");

    const [deviceGroups, setDeviceGroups] = useState<any[]>([]);
    const [selectedCoord, setSelectedCoord] = useState<string>("");
    const [devicesInGroup, setDevicesInGroup] = useState<any[]>([]);

    const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
    const [existingMeasurements, setExistingMeasurements] = useState<any[]>([]);

    const [message, setMessage] = useState<string | null>(null);

    // --- Effects ---
    useEffect(() => {
        fetchSamples();
    }, []);

    useEffect(() => {
        if (selectedDeviceId) {
            fetchMeasurements(selectedDeviceId);
        } else {
            setExistingMeasurements([]);
        }
    }, [selectedDeviceId]);

    // --- Fetchers ---
    const fetchSamples = async () => {
        const res = await fetch("http://localhost:8000/api/data/samples");
        if (res.ok) setSamples(await res.json());
    };

    const fetchDeviceGroups = async (sampleId: string) => {
        const res = await fetch(`http://localhost:8000/api/data/samples/${sampleId}/device-groups`);
        if (res.ok) setDeviceGroups(await res.json());
    };

    const fetchMeasurements = async (deviceId: string) => {
        const res = await fetch(`http://localhost:8000/api/data/samples/${selectedSampleId}/devices/${deviceId}/measurements`);
        if (res.ok) setExistingMeasurements(await res.json());
    };

    // --- Handlers ---
    const handleSampleSelect = (id: string) => {
        setSelectedSampleId(id);
        setSelectedCoord("");
        setDevicesInGroup([]);
        setSelectedDeviceId("");
        if (id) fetchDeviceGroups(id);
    };

    const handleGroupSelect = (coordStr: string) => {
        setSelectedCoord(coordStr);
        setSelectedDeviceId("");
        if (coordStr) {
            const group = deviceGroups.find(g => JSON.stringify(g.coord) === coordStr);
            if (group) setDevicesInGroup(group.devices);
        }
    };

    // Refresh groups and keep selection if possible
    const refreshGroups = async () => {
        if (!selectedSampleId) return;
        await fetchDeviceGroups(selectedSampleId);
        // We need to update devicesInGroup if the currently selected group changed
        if (selectedCoord) {
            const res = await fetch(`http://localhost:8000/api/data/samples/${selectedSampleId}/device-groups`);
            if (res.ok) {
                const groups = await res.json();
                // setDeviceGroups(groups); // Already done by fetchDeviceGroups? No, fetchDeviceGroups does it.
                // But fetchDeviceGroups is async, so we might need to wait or rely on state update.
                // Actually fetchDeviceGroups sets state.
                // But in this logic, we are re-fetching.
                // Let's rely on fetchDeviceGroups which sets state `deviceGroups`.
                // But `devicesInGroup` is separate state. We need to update it.

                const group = groups.find((g: any) => JSON.stringify(g.coord) === selectedCoord);
                if (group) setDevicesInGroup(group.devices);
                else setDevicesInGroup([]);
            }
        }
    };

    const handleDeleteDevice = async (deviceId: string) => {
        if (!confirm(`Delete device ${deviceId}?`)) return;
        try {
            const res = await fetch(`http://localhost:8000/api/data/samples/${selectedSampleId}/devices/${deviceId}`, { method: "DELETE" });
            if (!res.ok) throw new Error((await res.json()).detail);
            await refreshGroups();
        } catch (e: any) {
            alert("Error: " + e.message);
        }
    };

    const handleDeleteMeasurement = async (measId: string) => {
        if (!confirm("Delete measurement?")) return;
        try {
            const res = await fetch(`http://localhost:8000/api/data/measurements/${measId}`, { method: "DELETE" });
            if (!res.ok) throw new Error((await res.json()).detail);

            fetchMeasurements(selectedDeviceId);
        } catch (e: any) {
            alert("Error: " + e.message);
        }
    };

    const handleRegisterMeasurement = async (data: any) => {
        const res = await fetch("http://localhost:8000/api/data/measurements", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Registration failed");
        }

        const resData = await res.json();
        setMessage(`Success! Measurement ID: ${resData.id}`);
        fetchMeasurements(selectedDeviceId);
    };

    return (
        <div className="flex flex-col items-center min-h-screen p-8 bg-gray-50">
            <div className="w-full max-w-4xl flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold text-gray-800">Register Data (Refactored)</h1>
                <Link href="/" className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">Back</Link>
            </div>

            <div className="w-full max-w-4xl bg-white p-8 rounded-xl shadow-lg border border-gray-200 space-y-8">

                <SampleSection
                    samples={samples}
                    selectedSampleId={selectedSampleId}
                    onSelect={handleSampleSelect}
                    onRefresh={fetchSamples}
                />

                <DeviceGroupSection
                    sampleId={selectedSampleId}
                    selectedCoord={selectedCoord}
                    deviceGroups={deviceGroups}
                    devicesInGroup={devicesInGroup}
                    onSelect={handleGroupSelect}
                    onRefreshGroups={refreshGroups}
                    onDeleteDevice={handleDeleteDevice}
                />

                <DeviceSection
                    selectedCoord={selectedCoord}
                    devicesInGroup={devicesInGroup}
                    selectedDeviceId={selectedDeviceId}
                    onSelect={setSelectedDeviceId}
                />

                <MeasurementSection
                    sampleId={selectedSampleId}
                    deviceId={selectedDeviceId}
                    existingMeasurements={existingMeasurements}
                    onRegister={handleRegisterMeasurement}
                    onDeleteMeasurement={handleDeleteMeasurement}
                    message={message}
                />

            </div>
        </div>
    );
}
