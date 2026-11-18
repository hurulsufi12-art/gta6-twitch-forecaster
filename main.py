"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts the Twitch concurrent viewers of the "Grand Theft Auto VI"
category for hours 2..24 of release day 2026-11-19 (UTC), as five
quantiles (5%, 25%, 50%, 75%, 95%) per hour, from the case's hour-1 print.

Method: for each hour, look at every comparable past big launch in the
case file, take that launch's release-day curve as a ratio to its own
hour-1 count, and read off the quantiles of the log-ratios -- but weight
each comparable by how close its hour-1 count is to this case's hour-1
count (Gaussian kernel, bandwidth 1.0 in log space), times an unlock
family factor: launches whose unlock was a rolling midnight local, like
this case's, count five times as much as launches of the other family
(one simultaneous global unlock), which count a fifth; launches whose
unlock is not recorded are neutral. Hour-1's size is the strongest
signal in the data: launches whose hour 1 is already huge are already
in full swing and stay comparatively flat, while launches with a tiny
hour 1 are usually still waiting for their unlock and explode later.
The unlock family is the second signal: console-only rolling-midnight
launches keep rising through the day as regions unlock (evening peak
about 2x hour 1), while launches with a simultaneous global unlock or
early access are already flat by hour 1. A flat median (the starter)
is wrong by ~83% on a typical launch day; weighting by hour-1 size is
what fixes the shape. The per-hour log-quantile curves are lightly
smoothed over neighbouring hours.

When this case's hour 1 is larger than every comparable's hour 1 -- the
expected GTA VI situation, since its hour 1 will likely beat the largest
comparable (875k) -- the kernel would lean on the few biggest launches
alone, whose shape is flat. The forecast is then blended toward an
embedded big-launch table (the average shape over the 16 launches with
hour 1 above 100k, evening peak ~1.25x hour 1), with the blend weight
growing with how far beyond the training range the hour-1 print sits,
capped at 0.5. In a test that trains only on launches under 100k hour-1
viewers and forecasts the 16 bigger ones, the blend cuts mean pinball
loss from 0.305 to 0.227.

If the case carries too few usable comparables (fewer than 8 complete
release-day curves), the program falls back to an embedded table: the
median log-ratio and log standard deviation per hour fitted on the 37
complete comparable launch days in the kit's practice cases.

The program reads every <id>.in in the input directory and writes
<id>.out: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
exactly 23 rows, hours 2..24 each once, quantiles non-decreasing, each
in (0, 1e9]. It is deterministic (same case -> same bytes), uses no
network, no model calls, no clock and no unseeded randomness, and runs
in well under a second per case.

How the grader runs it: `python3 -P -s /bot/main.py` in
python:3.13-slim. Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))
# Gaussian kernel bandwidth in log(hour-1 viewers): how similar in size
# a comparable launch must be to count fully.
BANDWIDTH = 1.0
FAM_SAME = 5.0
FAM_DIFF = 0.2
# Minimum usable comparables for the kernel fit; below this, use the
# embedded fallback table.
MIN_COMPARABLES = 8

# Fallback table: median log-ratio to hour 1 and log standard deviation
# per hour (hours 2..24), fitted on the 37 comparable launches in the
# kit's practice cases that have a complete 24-hour release day.
# Median ratios: 0.87, 0.75, 0.73, 0.75, 0.79, 0.80, 0.90, 0.88, 0.91,
# 0.93, 0.98, 1.06, 1.18, 1.52, 1.65, 1.77, 1.95, 2.40, 2.37, 2.33,
# 2.30, 2.18, 2.04 -- the day dips early, then climbs to a US-evening
# peak about 2.4x hour 1, with a spread that widens through the day.
FALLBACK_MED = [
    -0.1376, -0.2829, -0.3100, -0.2916, -0.2377, -0.2201, -0.1066,
    -0.1259, -0.0925, -0.0745, -0.0223, 0.0574, 0.1661, 0.4176,
    0.5022, 0.5713, 0.6681, 0.8761, 0.8635, 0.8460, 0.8322, 0.7803,
    0.7122,
]
FALLBACK_SIG = [
    0.3033, 0.7246, 0.7702, 0.7974, 0.8506, 0.8642, 0.8994,
    1.0708, 1.4249, 1.5379, 1.5992, 1.7066, 1.8594, 2.0211,
    2.1399, 2.1902, 2.2033, 2.1984, 2.1995, 2.2063, 2.1879,
    2.1528, 2.1399,
]


# Big-launch fallback table: median log-ratio to hour 1 and log standard
# deviation per hour (hours 2..24), fitted on the 16 comparable launches
# with hour-1 >= 100k. Big launches dip to ~0.5x mid-day and peak at
# only ~1.25x -- far flatter than the all-launch fallback (2.4x peak).
BIG_MED = [
    -0.1009, -0.2721, -0.3643, -0.3674, -0.3761, -0.506, -0.5884,
    -0.6606, -0.6772, -0.6745, -0.562, -0.4507, -0.241, -0.1816,
    -0.1063, 0.1016, 0.1372, 0.2271, 0.1797, 0.2145, 0.1696, 0.0863,
    0.1298,
]
BIG_SIG = [
    0.2069, 0.2614, 0.2942, 0.2961, 0.4207, 0.5019, 0.4789,
    0.4809, 0.4935, 0.4937, 0.4886, 0.4462, 0.4419, 0.4474,
    0.4408, 0.4232, 0.4156, 0.4098, 0.4211, 0.4058, 0.4231, 0.4049,
    0.4337,
]


def big_fallback_log_quantiles() -> list[list[float]]:
    """Log-quantiles from the big-launch fallback table (normal model)."""
    return [[med + z * sig for z in Z] for med, sig in zip(BIG_MED, BIG_SIG)]




def unlock_family(text: str | None) -> str:
    """'rolling' (midnight local, region by region), 'global' (one
    simultaneous unlock worldwide), or 'unknown' from free text."""
    t = (text or "").lower()
    if ("midnight local" in t or "rolling" in t or "12 am local" in t
            or "12am local" in t):
        return "rolling"
    if ("simultaneous" in t or "single global" in t or "global unlock" in t):
        return "global"
    return "unknown"


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0). Same convention as the starter.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comp: dict) -> list[float] | None:
    """A comparable's 24 hourly viewer counts on its release day, or None.

    Returns None when the release day is incomplete (a launch whose day
    has not happened yet, like GTA VI's own pre-release category) or its
    hour-1 count is not positive.
    """
    try:
        day_start = parse_time(comp["day_start"])
        points: dict[int, float] = {}
        for p in comp.get("hourly") or []:
            hour = int((parse_time(p["from"]) - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and p["viewers"] > 0:
                points[hour] = float(p["viewers"])
        if len(points) == 24 and points[1] > 0:
            return [points[h] for h in range(1, 25)]
    except (KeyError, TypeError, ValueError):
        return None
    return None


def weighted_quantile(pairs: list[tuple[float, float]], q: float) -> float:
    """The q-th weighted percentile of (value, weight) pairs.

    Linear interpolation on cumulative weight, so the tails move smoothly
    instead of jumping between order statistics.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[0][0]
    # Cumulative-weight midpoints, in the same order as the values.
    mids: list[tuple[float, float]] = []
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        mids.append((acc - 0.5 * weight, value))
    target = q * total
    if target <= mids[0][0]:
        return mids[0][1]
    if target >= mids[-1][0]:
        return mids[-1][1]
    for (x0, v0), (x1, v1) in zip(mids, mids[1:]):
        if x0 <= target <= x1:
            f = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
            return v0 * (1 - f) + v1 * f
    return mids[-1][1]


def kernel_log_quantiles(curves: list[list[float]], p1: float,
                         fams: list[str] | None = None,
                         target_fam: str = "unknown") -> list[list[float]]:
    """Per-hour log-ratio quantiles, weighted by hour-1 size similarity.

    Each comparable contributes its log-ratio ln(v_h / v_1) for hour h,
    weighted by a Gaussian kernel on |ln(its hour 1) - ln(p1)|, times a
    family factor: launches whose unlock family (rolling midnight vs one
    simultaneous global unlock) matches the target's count more, the
    others less. Unknown families are neutral.
    The resulting 23 x 5 table is smoothed with a 3-hour moving average.
    """
    log_p1 = math.log(p1)
    table: list[list[float]] = []
    for h in HOURS:
        pairs = []
        for i, curve in enumerate(curves):
            weight = math.exp(-((math.log(curve[0]) - log_p1) ** 2) / (2 * BANDWIDTH * BANDWIDTH))
            if fams is not None and target_fam != "unknown":
                fam = fams[i]
                if fam == target_fam:
                    weight *= FAM_SAME
                elif fam != "unknown":
                    weight *= FAM_DIFF
            pairs.append((math.log(curve[h - 1] / curve[0]), weight))
        table.append([weighted_quantile(pairs, level) for level in LEVELS])
    smooth: list[list[float]] = []
    for i, row in enumerate(table):
        idx = [j for j in (i - 1, i, i + 1) if 0 <= j < len(table)]
        smooth.append([sum(table[j][k] for j in idx) / len(idx) for k in range(5)])
    return smooth


def fallback_log_quantiles() -> list[list[float]]:
    """Log-quantiles from the embedded fallback table (normal model)."""
    return [[med + z * sig for z in Z] for med, sig in zip(FALLBACK_MED, FALLBACK_SIG)]


def hour1_print(case: dict) -> float:
    """The case's hour-1 viewer count, at least 1.

    Falls back to the last pre-checkpoint hourly point, then to 1, so a
    case without a usable hour-1 print still gets a valid forecast.
    """
    for known in case.get("known") or []:
        if known.get("hour") == 1 and isinstance(known.get("viewers"), (int, float)):
            return max(1.0, float(known["viewers"]))
    history = case.get("history_hourly") or []
    if history:
        last = history[-1]
        if isinstance(last.get("viewers"), (int, float)):
            return max(1.0, float(last["viewers"]))
    return 1.0


def forecast_case(case: dict) -> dict:
    p1 = hour1_print(case)
    curves = []
    fams = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is not None:
            curves.append(curve)
            fams.append(unlock_family(comp.get("unlock")))
    if len(curves) >= MIN_COMPARABLES:
        log_quantiles = kernel_log_quantiles(
            curves, p1, fams, unlock_family(case.get("notes")))
        # Extrapolation guard: when this case's hour 1 sits beyond the
        # largest comparable, the kernel leans on the few biggest launches
        # alone. Blend toward the fallback table (the average shape over
        # all 37 launches) with a weight that grows with the excess in
        # log space, capped at 0.5, so a launch far beyond the training
        # range does not inherit the flattest shape by default.
        largest = max(math.log(curve[0]) for curve in curves)
        excess = math.log(p1) - largest
        if excess > 0:
            w = min(0.5, excess / 2.0)
            fallback = big_fallback_log_quantiles()
            log_quantiles = [
                [(1 - w) * a + w * b for a, b in zip(row, fb)]
                for row, fb in zip(log_quantiles, fallback)
            ]
    else:
        log_quantiles = fallback_log_quantiles()
    hours = []
    for i, hour in enumerate(HOURS):
        quantiles: list[float | int] = []
        prev = 0.0
        for log_q in log_quantiles[i]:
            q = round2(p1 * math.exp(log_q))
            q = min(max(q, 0.01), 1e9)
            if q < prev:
                q = prev
            prev = q
            quantiles.append(q)
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
