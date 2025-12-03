import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

@dataclass
class ParsedIVSeries:
    id_mA: List[float]
    vd_mV: List[float]
    r_ohm: List[float]
    warnings: List[str]

def load_iv_data(file_path: str) -> ParsedIVSeries:
    """
    別形式のIVファイルを読み込み、ParsedIVSeries を返す。

    仕様:
    - 先頭2行は無視
    - 3行目以降を1行ずつ読み取る
    - 各行は7つの値(空白区切り)を想定し、Id(A)=1列目, Vd(V)=2列目, R(ohm)=5列目
    - IdとVdはm単位に変換(×1000)して格納
    - 空行や非数値行(例: "= = = ="やコメント)は自動でスキップ
    """
    id_milliamp: list[float] = []
    vd_millivolt: list[float] = []
    resistance_ohm: list[float] = []
    warnings: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_index, raw_line in enumerate(f, start=1):
            if line_index <= 2:
                continue
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()  # 空白(スペース/タブ)の複数区切りに対応
            if len(parts) < 5:
                warnings.append(f"line {line_index}: 列数不足のためスキップ")
                continue

            try:
                id_amp = float(parts[0])
                vd_volt = float(parts[1])
                r_ohm = float(parts[4])
            except ValueError:
                # 非数値行はスキップ
                warnings.append(f"line {line_index}: 数値変換不可のためスキップ")
                continue

            id_milliamp.append(id_amp * 1_000.0)
            vd_millivolt.append(vd_volt * 1_000.0)
            resistance_ohm.append(r_ohm)

    return ParsedIVSeries(
        id_mA=id_milliamp,
        vd_mV=vd_millivolt,
        r_ohm=resistance_ohm,
        warnings=warnings,
    )