# GTA VI Launch-Day Twitch Forecaster

A program that forecasts how many people will be watching Grand Theft
Auto VI live on Twitch in each hour of its release day, 2026-11-19
(UTC), for make.fun
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
— five quantiles (5% / 25% / 50% / 75% / 95%) per hour, hours 2–24.

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hurulsufi12-art/gta6-twitch-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hurulsufi12-art/gta6-twitch-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

## How it works

- Reads the case's hour-1 viewer count, then looks at every comparable
  past big launch shipped in the case file and takes each launch's
  release-day curve as a ratio to its own hour-1 count.
- For each hour, the quantiles of those log-ratios are read off with
  weights from a Gaussian kernel on hour-1 size: launches whose hour 1
  was about as big as this case's count most. Hour-1 size is the
  strongest signal in the data — a huge hour 1 means the launch is
  already in full swing and stays comparatively flat, while a tiny hour 1
  usually means the unlock is still coming and the audience explodes
  later. The weights are multiplied by an unlock-family factor: a
  comparable whose unlock was a rolling midnight local, like GTA VI's,
  counts five times as much as one with a simultaneous global unlock,
  which counts a fifth (unknown unlocks are neutral). The log-quantile
  curves are lightly smoothed over neighbouring hours.
- When this case's hour 1 sits beyond every comparable's, the forecast
  is blended toward an embedded big-launch table (the 16 launches with
  hour 1 above 100k, evening peak about 1.25x), so a launch far beyond
  the training range does not inherit the flattest shape by default.
- If the case carries too few usable comparables, it falls back to an
  embedded table (median log-ratio and log spread per hour, fitted on
  the 37 complete comparable launch days in the kit's practice cases).
- Deterministic, standard library only, no network, no model calls, no
  clock: the same case always produces the same bytes, in well under a
  second.

## Validation

Leave-one-out over the 37 complete comparable launch days in the kit:
mean pinball loss on the log scale **0.389**, against **0.930** for the
flat starter baseline (and 0.174 vs 0.254 on the ten most GTA-like
launches: rolling-midnight unlocks with a large hour 1). Coverage of
the 90% band is about 0.87; the median forecast misses by about 46%.

## Files

- `main.py` — the entrypoint (Python 3.13, standard library only).
- `LICENSE` — MIT.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
