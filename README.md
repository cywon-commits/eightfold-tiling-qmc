# eightfold-tiling-qmc

Code and data for *Quantum selection of eightfold rhombus tilings: faceting at zero field and a charge-budget magnetization staircase* (C. Won). The spin-1/2 Heisenberg antiferromagnet lives on eightfold (Ammann–Beenker type) rhombus tilings. The study uses exact combinatorics, linear spin-wave theory (LSWT), and sign-free SSE quantum Monte Carlo.

## Layout

| folder | contents |
|---|---|
| `sse/` | Deterministic-loop SSE (`sse_det.py`), a version that splits exchange into normal/in-plane parts and local moments (`sse_split.py`), sector exact diagonalization (`ed_sector.py`) and the validation script (`validate.py`). |
| `lswt/` | LSWT energies (`lswt.py`), local moments and bond correlators (`lswt_local.py`), the Hartree (Oguchi) correction and ratio `r_H` (`hartree.py`, `all_rH.py`), and the canted square lattice in a field (`esq_h.py`). |
| `qmc_stage1/`–`qmc_stage3/` | Earlier QMC stages (October 2026): zero-field quintet, crystals, ribbons, the QMC-calibrated staircase, and blind test I of the Hartree ratio. |
| `sector_search/` | Enumeration of all winding sectors of periodic tilings with N ≤ 24 tiles per cell, plus the flip enumeration, screening, field linear program, resampling, and Fig. 7 data. |
| `qmc_stage4_seeds/` | Seeded QMC (3 seeds, β = 300, Lieb sector) of every phase on the ground-state paths, with the blind test II table. |
| `manuscript/` | LaTeX source, figures, and figure scripts. |

## Sector search (`sector_search/`)

- `sectors.py N` enumerates rank-2 lattices of lifted periods (T1, T2) in Z⁴ whose Plücker minors all have one sign, reduced by GL2(Z) × D8. Results for N ≤ 24 are in `data/sectors_N24.json` (5761 sectors). `check_small.py` cross-checks N ≤ 3 by brute force.
- `build.py` holds the general de Bruijn construction and the flip BFS (cap 3000 tilings per sector). `run_enum.py N0 N1 CAP [k/K]` runs the enumeration.
- The raw enumeration files (≈0.7 GB, with all tilings) are not stored. `data/enum_summary_N24.json.gz` keeps, for every sector, the period matrix, the Plücker vector, whether the BFS was exhausted, the number of tilings, and the lowest screening energy for each composition (σ, τ). `run_enum.py` regenerates the full files.
- `analyze_enum.py N` screens the tilings and writes `data/screen_N*.json`: LSWT at about 600 and 1500 sites, plus the Hartree ratio. `compare.py N` merges these with the earlier phase set (`data/compare_N*.json`).
- `resample.py 4000 compare_N*.json out.json` gives the QMC-anchored staircase and the presence probability of each plateau over 4000 redraws. Phase energies use the same-cell QMC/LSWT ratio times the converged LSWT cost, and the finite-size change between the two largest cells is added to the error. Outputs are in `data/resample_*.json`.
- `newpath.py` writes the central ground-state paths (`data/path_N*.json`). `window.py` lists the phases on the paths and the energy gain per search depth.
- `data/window_crystals_W1-W31.json` lists the window crystals of Appendix E (Fig. 10, Table II) under their labels W1–W31. For each crystal it gives the lifted periods, the tiles (corner lifts and families), the projection used for drawing, and the numbers shown in Table II. `internal_name` is the working label used in the run files.
- `approx_sectors.py S R` and `run_approx.py` cover the targeted check of the σ = 12/17 sectors (N = 29). All 7×10⁴ tilings have M = 0, as required by the parity argument in the paper.

## Seeded QMC (`qmc_stage4_seeds/`)

- `runs/KEY_L_sN.json` holds one run per phase, supercell size L, and seed. Each file stores energies, normal and in-plane exchange, bins, wall time, R and σ. The scripts `qseeds*.py` define the job lists of the successive rounds. Each round added the phases that had entered a ground-state path, until every path phase had been simulated (40 phases).
- `newphases.json` gives the structure of every phase found by the sector search (tiles, periods) together with its LSWT cost and the Hartree-predicted ratio. `qmc_seed_table.json` gives the final QMC cost per rhombus.
- `blind_test_II.csv` lists, for 31 phases, the ratio predicted with Eq. (15) of the paper and the measured ratio. The predictions were computed during screening (`data/screen_N16/21/24.json`, written 2026-11-19 01:12, 02:10 and 05:34 KST) before any QMC run of the phase; the last column gives the time of the first QMC file of each phase. All of these files were committed together on 2026-11-19, so the commit date does not separate prediction from measurement; the file times in the table do.

## Paths

The scripts were written for the working directories of the original sessions and contain absolute paths such as `/mnt/user-data/outputs/...` and `/home/claude/fieldfig`. Adjust these paths, or create symbolic links, before rerunning. Python 3 with numpy, scipy, and numba is required.

## Use of AI

Claude (Anthropic; Claude Opus 5.5, November 2026) was used to derive and check analytic results, to write the analysis and plotting code, to run numerical tests and to draft and edit the text.
