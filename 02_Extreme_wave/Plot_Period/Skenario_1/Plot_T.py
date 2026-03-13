import os
import re
import numpy as np
import pandas as pd

# ==========================
# EDIT DI SINI SAJA
# ==========================
RUN_DIR  = r"D:\ANCOL\Hidrodinamika\Extreme_wave\DWL_50\Skenario_1\WNW\10000"
SP1_FILE = "skenario-1.locn3.sp1"

OUT_DIR  = r"D:\ANCOL\Hidrodinamika\Extreme_wave\Plot_Period"
OUT_CSV  = "WNW_RP10000_locn3_period.csv"   # bebas, kasih nama sesuai arah/RP
# ==========================


def parse_swan_sp1(sp1_path: str):
    with open(sp1_path, "r", errors="replace") as f:
        lines = f.readlines()

    # LOCATIONS
    i_loc = next(i for i, l in enumerate(lines) if l.strip().startswith("LOCATIONS"))
    nloc = int(lines[i_loc + 1].split()[0])
    coords = []
    for j in range(nloc):
        x, y = lines[i_loc + 2 + j].split()[:2]
        coords.append((float(x), float(y)))
    coords = np.array(coords, dtype=float)

    # AFREQ
    i_af = next(i for i, l in enumerate(lines) if l.strip().startswith("AFREQ"))
    nfreq = int(lines[i_af + 1].split()[0])
    freqs = np.array([float(lines[i_af + 2 + k].split()[0]) for k in range(nfreq)], dtype=float)

    # QUANT (exception)
    i_q = next(i for i, l in enumerate(lines) if l.strip().startswith("QUANT"))
    nq = int(lines[i_q + 1].split()[0])

    exceptions = []
    cursor = i_q + 2
    for _ in range(nq):
        cursor += 1  # quantity name
        cursor += 1  # unit
        exceptions.append(float(lines[cursor].split()[0]))
        cursor += 1
    exc0 = exceptions[0] if exceptions else None

    # TIME line
    date_idx = next(i for i in range(cursor, len(lines)) if re.match(r"^\d{8}\.\d{6}", lines[i].strip()))
    cursor = date_idx + 1

    # DATA
    E = np.zeros((nloc, nfreq), dtype=float)
    for loc_id in range(1, nloc + 1):
        while cursor < len(lines) and lines[cursor].strip() == "":
            cursor += 1
        if not lines[cursor].startswith("LOCATION"):
            raise ValueError(f"Expected 'LOCATION' at line {cursor+1}, got: {lines[cursor][:80]!r}")
        cursor += 1

        for k in range(nfreq):
            v = float(lines[cursor + k].split()[0])
            if exc0 is not None and abs(v - exc0) < 1e-9:
                v = 0.0
            E[loc_id - 1, k] = v
        cursor += nfreq

    return coords, freqs, E


def compute_periods(freqs: np.ndarray, E: np.ndarray):
    f = freqs
    m0  = np.trapz(E, f, axis=1)
    m2  = np.trapz((f**2)[None, :] * E, f, axis=1)
    m_1 = np.trapz((1.0 / f)[None, :] * E, f, axis=1)

    TM02  = np.sqrt(np.divide(m0, m2, out=np.full_like(m0, np.nan), where=m2 > 0))      # Tm2,0
    TMM10 = np.divide(m_1, m0, out=np.full_like(m0, np.nan), where=m0 > 0)               # Tm-1,0
    fp = f[np.argmax(E, axis=1)]
    Tp = np.divide(1.0, fp, out=np.full_like(fp, np.nan), where=fp > 0)                  # peak period

    return Tp, TM02, TMM10


def main():
    sp1_path = os.path.join(RUN_DIR, SP1_FILE)
    if not os.path.isfile(sp1_path):
        raise FileNotFoundError(f"File tidak ketemu: {sp1_path}")

    coords, freqs, E = parse_swan_sp1(sp1_path)
    Tp, TM02, TMM10 = compute_periods(freqs, E)

    df = pd.DataFrame({
        "loc_id": np.arange(1, coords.shape[0] + 1),
        "XP": coords[:, 0],
        "YP": coords[:, 1],
        "Tp_s": Tp,
        "Tm2_0_TM02_s": TM02,
        "Tm_1_0_TMM10_s": TMM10,
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_CSV)
    df.to_csv(out_path, index=False)

    print("OK ->", out_path)
    print(df.head(5))


if __name__ == "__main__":
    main()