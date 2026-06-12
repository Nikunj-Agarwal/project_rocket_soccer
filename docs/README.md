<!--
DOC PLACEHOLDERS — replace after each pipeline run
──────────────────────────────────────────────────
{LATEST_INTEGRATION_BATCH}   Newest folder under data/tests/integration/ (YYYYMMDD_HHMMSS).
                             python -c "from src.data_layout import latest_integration_batch; b=latest_integration_batch(); print(b.name if b else 'NONE')"
{LATEST_COMPARISON_RUN}      Newest folder under data/tests/comparison/.
{PREVIOUS_INTEGRATION_BATCH} Pre dual-model reference batch for "previous try" narrative (example on disk: 20260612_155705).
Metrics marked *fill from batch* → summary.json, batch.log, or comparison.csv under those dirs.
-->

# Project Documentation

Read these documents to understand the architecture, parameters, and pipelines of the robot soccer striker:

| Document | Contents |
| :--- | :--- |
| **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** | End-to-end architecture: dual StrikeNet variants, three planner modes, offline training, online two-phase loop, comparison harness. |
| **[PHYSICS_CONSTRAINTS_ASSUMPTIONS.md](PHYSICS_CONSTRAINTS_ASSUMPTIONS.md)** | Field dimensions, elastic/inelastic collision models, NMPC bounds, warm-start, target offset, reachability model. |
| **[PIPELINE_LOGIC.md](PIPELINE_LOGIC.md)** | Dataset generation, dual-variant training, `decide_strike_target()`, shrinking-horizon NMPC, testing & comparison pipelines. |
| **[DATA_AND_REPORTS.md](DATA_AND_REPORTS.md)** | Output formats, batch/comparison directory layout, metadata schema, linking plots to raw runs. |
| **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** | Formal research paper: abstract, methodology, **previous try vs current system**, results placeholders, references. |
| **[LITERATURE_REVIEWS.md](LITERATURE_REVIEWS.md)** | Categorized academic references supporting the project design. |
| **[UPDATE.md](UPDATE.md)** | Phase 5 overhaul, evaluation-validity fixes, and dual-model + 3-way comparison upgrade. |
| **[PHYSICS_INFORMED_PREDICTION.md](PHYSICS_INFORMED_PREDICTION.md)** | Physics-informed (structured) StrikeNet: design rationale and how it is evaluated against the legacy variant. |
| **[legacy/](legacy/)** | **[ARCHIVE]** Pre–Phase 5 documentation snapshots. |
| **[legacy_2/](legacy_2/)** | **[ARCHIVE]** Post evaluation-validity sync snapshot (Issues 1–3). |

For operational commands, see the root [README.md](../README.md) and `run_pipeline.ps1` / `run_pipeline.sh` (7-step full pipeline).
