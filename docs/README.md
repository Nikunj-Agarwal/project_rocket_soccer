<!--
DOC PLACEHOLDERS — replace after each pipeline run
──────────────────────────────────────────────────
{LATEST_INTEGRATION_BATCH}   Newest folder under data/tests/integration/ (YYYYMMDD_HHMMSS).
                             python -c "from src.data_layout import latest_integration_batch; b=latest_integration_batch(); print(b.name if b else 'NONE')"
{LATEST_COMPARISON_RUN}      Newest folder under data/tests/comparison/.
{PREVIOUS_INTEGRATION_BATCH} Pre dual-model reference batch (example: 20260612_155705).

Reference comparison run with corrected deployed latency (20260613_025809):
  data/reports/plots/comparison/20260613_025809/worth_it_summary.md

Metrics marked *fill from batch* → summary.json, batch.log, comparison.csv, or worth_it_summary.md.
-->

# Project Documentation

Read these documents to understand the architecture, parameters, and pipelines of the robot soccer striker:

| Document | Contents |
| :--- | :--- |
| **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** | End-to-end architecture: dual StrikeNet variants, three planner modes, hybrid fallback vs full analytic, latency model, comparison harness. |
| **[PHYSICS_CONSTRAINTS_ASSUMPTIONS.md](PHYSICS_CONSTRAINTS_ASSUMPTIONS.md)** | Field dimensions, elastic/inelastic collision models, NMPC bounds, warm-start, target offset, reachability model. |
| **[PIPELINE_LOGIC.md](PIPELINE_LOGIC.md)** | Dataset generation, dual-variant training, `decide_strike_target()`, shrinking-horizon NMPC, testing & 8-step pipeline. |
| **[DATA_AND_REPORTS.md](DATA_AND_REPORTS.md)** | Output formats, batch/comparison directory layout, metadata schema, latency fields, linking plots to raw runs. |
| **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** | Formal research paper: abstract, methodology, results, latency/cost-benefit argument, references. |
| **[LITERATURE_REVIEWS.md](LITERATURE_REVIEWS.md)** | Categorized academic references supporting the project design. |
| **[UPDATE.md](UPDATE.md)** | Phase 5 overhaul, evaluation-validity fixes, dual-model + 3-way comparison, deployed latency measurement. |
| **[PHYSICS_INFORMED_PREDICTION.md](PHYSICS_INFORMED_PREDICTION.md)** | Structured StrikeNet: design rationale, empirical findings, evaluation caveats. |
| **[legacy/](legacy/)** | **[ARCHIVE]** Pre–Phase 5 documentation snapshots. |
| **[legacy_2/](legacy_2/)** | **[ARCHIVE]** Post evaluation-validity sync snapshot (Issues 1–3). |

For operational commands, see the root [README.md](../README.md) and `run_pipeline.ps1` / `run_pipeline.sh` (8-step eval pipeline; skip data/train with `-SkipData -SkipTrain`).
