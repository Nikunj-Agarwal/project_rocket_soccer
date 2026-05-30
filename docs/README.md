# Project Documentation

Read these documents to understand the architecture, parameters, and pipelines of the robot soccer striker:

| Document | Contents |
| :--- | :--- |
| **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** | End-to-end architecture, offline training, and online two-phase loops. |
| **[PHYSICS_CONSTRAINTS_ASSUMPTIONS.md](PHYSICS_CONSTRAINTS_ASSUMPTIONS.md)** | Field dimensions, elastic/inelastic collision models, NMPC bounds, warm-start, and target offset equations. |
| **[PIPELINE_LOGIC.md](PIPELINE_LOGIC.md)** | Step-by-step algorithms for dataset generation, training representations, online shrinking-horizon NMPC, and active braking. |
| **[DATA_AND_REPORTS.md](DATA_AND_REPORTS.md)** | Output data formats, batch directories structure, and linking results to reporting plots. |
| **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** | Formal research paper: abstract, methodology, results, and references. |
| **[LITERATURE_REVIEWS.md](LITERATURE_REVIEWS.md)** | Categorized academic references supporting the project design. |
| **[UPDATE.md](UPDATE.md)** | Overview of the Phase 5 (Strike & Score) overhaul, including key design decisions, fixes, and validation benchmarks. |
| **[legacy/](legacy/)** | **[ARCHIVE]** Folder containing historic documentation versions prior to Phase 5. |

For operational commands, see the root [README.md](../README.md).
