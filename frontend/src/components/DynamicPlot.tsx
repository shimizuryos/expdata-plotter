"use client";

import dynamic from "next/dynamic";
import { ComponentProps } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function DynamicPlot(props: ComponentProps<typeof Plot>) {
    return <Plot {...props} />;
}
