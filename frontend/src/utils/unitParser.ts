/**
 * Parses a comma-separated string of thicknesses with an optional trailing unit.
 * Converts specified length values to internal `nm` scale representations.
 * Available Units: nm, A (Angstrom), um, mm, m, min (minutes), V (Volts).
 * 
 * Example:
 *  "1, 2, 5 nm" -> [1.0, 2.0, 5.0]
 *  "10, 20 A" -> [1.0, 2.0]
 *  "0.1 um" -> [100.0]
 *  "10, 20 min" -> [10.0, 20.0]
 */
export function parseThicknessSequence(input: string): number[] {
    if (!input || input.trim() === "") return [];

    const str = input.trim();

    // Check if the last part is a text string (the unit)
    const match = str.match(/([a-zA-Z]+)$/);
    let unit = "nm"; // default
    let numbersStr = str;

    if (match) {
        unit = match[1];
        numbersStr = str.substring(0, str.length - unit.length).trim();
    }

    // Supported conversion mapping to nm (or 1.0 identity for generic/time units)
    const multipliers: { [key: string]: number } = {
        "A": 0.1,
        "nm": 1.0,
        "um": 1e3,
        "mm": 1e6,
        "m": 1e9,
        "min": 1.0,
        "V": 1.0
    };

    if (!(unit in multipliers)) {
        throw new Error(`Unsupported unit '${unit}'. Supported units are: A, nm, um, mm, m, min, V`);
    }

    const m = multipliers[unit];

    // Split by comma, ensuring values map clean
    const stringVals = numbersStr.split(",").map(v => v.trim()).filter(v => v !== "");

    if (stringVals.length === 0) {
        throw new Error("No numbers provided before the unit.");
    }

    const numericVals: number[] = [];
    for (const s of stringVals) {
        const val = parseFloat(s);
        if (isNaN(val)) {
            throw new Error(`Invalid number '${s}' found in sequence.`);
        }
        numericVals.push(val * m);
    }

    return numericVals;
}
