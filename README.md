# SACER-RAG

## Overview

SACER-RAG is a long-document retrieval-augmented generation architecture designed to preserve coherent evidence when graph-supported context is incomplete, ambiguous, or at risk of collapse. The repository exposes one canonical full-model configuration and the command-line interfaces used for index construction, retrieval, answer generation, and evaluation.

This public repository contains the project structure, interfaces, configuration format, and non-confidential execution components. Selected core retrieval implementations are omitted from the public preprint release. The complete implementation is available to reviewers through a confidential package.

## Retrieval Context Collapse

Retrieval context collapse occurs when individually relevant chunks fail to preserve the local narrative, entity relationships, or supporting context required to answer a long-document question. SACER-RAG treats this as a routing problem: evidence is anchored to a persistent hierarchy, assessed for reliability, and either retained or reconstructed through a controlled branch of the full pipeline.

## Method Architecture

The method contains five named components:

- SACI constructs persistent entity-anchor-chunk mappings.
- AFER localizes anchors before contextual evidence is selected.
- ARG-R estimates anchor reliability and chooses a hierarchical route.
- CACR performs budget-constrained local evidence replacement for the medium-confidence branch.
- AECF selectively reconstructs candidate context for the low-confidence branch.

The graph structural signal is Boolean. It reports whether a valid graph structure exists; it does not independently supply a second retrieval pipeline.

## Full-Model Pipeline

The only supported retrieval flow is:

```text
SACI
-> graph structural signal
-> AFER
-> ARG-R
   high   -> retain AFER evidence
   medium -> CACR
   low    -> AECF
-> answer generation
```

Module-disable combinations, forced all-query routing, and public ablation profiles are not supported.

## Repository Structure

```text
configs/        Public full-model configuration template
evaluation/     Dataset-aware evaluation and reporting
indexing/       SACI construction interfaces and index loading utilities
model/          Retrieval, routing, generation, and dataset adapters
tests/          Public import, configuration, and redaction contracts
utils/          Configuration and generic file utilities
main.py         build-index, run, and evaluate command dispatcher
```

Datasets, model weights, generated indexes, answer files, caches, and historical experiment outputs are not distributed in the repository.

## Innovation-to-Source Mapping

| Innovation | Main source files | Responsibility |
| --- | --- | --- |
| SACI | `indexing/builder.py`, `indexing/loader.py` | Entity-anchor-chunk indexing and persistent mappings |
| Graph structural signal | `model/graph_signal.py`, `model/graphless_retriever.py` | Boolean structural-existence signal only |
| AFER | `model/anchor_localizer.py`, `model/anchor_candidate_filter.py`, `model/anchor_retrieval_adapter.py` | Anchor-first contextual evidence routing |
| ARG-R | `model/anchor_reliability_gate.py`, `model/gate_policy.py` | Reliability estimation and hierarchical routing |
| CACR | `model/context_collapse_risk_estimator.py`, `model/entity_anchor_router.py`, `model/prefiltered_anchor_localizer.py` | Budget-constrained local evidence replacement |
| AECF | `model/anchor_evidence_contextual_fallback.py` | Selective candidate-context reconstruction |
| Unified execution | `main.py`, `model/runner.py` | Index reuse, dataset/model dispatch and full-model execution |
| Automatic gate selection | `model/gate_policy.py`, `model/runner.py` | Public `gate` interface with internal standard/loose selection |

## Index Construction

`build-index` is the stable entry point for constructing and persisting the SACI hierarchy, dense sidecar data, anchor mappings, and validation metadata. Generic index loading, compatibility inspection, hashing, and file I/O remain public. The confidential SACI construction algorithm is not included in this release, so the public command raises a clear `NotImplementedError` when construction reaches that boundary.

## Retrieval and Routing

`run` preserves the full orchestration contract: inspect the requested index, build it if missing, reuse it when complete, execute the SACER-RAG pipeline, generate answers, and save records. The public release never silently substitutes dense retrieval, a baseline method, or a partial model. Execution stops with a clear confidentiality error when a redacted retrieval component is reached.

## Unified Gate Selection

The external configuration value is always:

```yaml
anchor_evidence_policy: gate
```

The runner selects the internal standard or loose gate from the configured dataset and model family. Direct gate-variant fields are rejected by configuration validation. Internal reliability thresholds, scoring expressions, and routing heuristics are not part of the public preprint release.

## Dataset Support

The loaders support InfiniteChoice, InfiniteQA, and NovelQA task formats. InfiniteQA remains open-ended question answering and is not converted to multiple-choice letter scoring. Users must obtain each dataset from its authorized source and configure a local placeholder path such as `/path/to/dataset`.

## Model Support

The execution layer supports Qwen and Llama model families through the common generation interface. Configure locally obtained model resources with placeholders such as `/path/to/qwen-model` or `/path/to/llama-model`; configure the embedding model as `/path/to/embedder`. No model weights are distributed here.

## Configuration

The release provides `configs/config.yaml` as the single full-model template. Required path values are placeholders and must be replaced locally. The `sacer` section accepts the canonical `gate` policy and the complete SACER-RAG flow only. Removed ablation names, module-disable switches, forced AECF routing, and manual gate variants are rejected.

## Commands

```bash
python main.py build-index --config configs/config.yaml
python main.py run --config configs/config.yaml
python main.py evaluate --config configs/config.yaml
```

Use `--dataset-name` to override `dataset.dataset_name` without changing the pipeline:

```bash
python main.py run --config configs/config.yaml --dataset-name InfiniteQALoader
```

## Output Files

With the confidential implementation installed, index artifacts are written under the configured index directory, answer records under the answer directory, and metrics under the evaluation directory. The public template uses repository-relative output locations. The public release alone does not produce paper-result artifacts because its core retrieval implementations are intentionally redacted.

## Public Release Scope

The public tree includes task loaders, configuration parsing and validation, generic index loading and inspection, generic model loading, answer formatting, file I/O, logging-compatible diagnostics, evaluation metrics, stable command entry points, public types, and module-level documentation. See `PUBLIC_RELEASE_SCOPE.md` for the exact omissions and guarantees.

## Confidential Reviewer Package

The confidential package contains the complete SACI, AFER, ARG-R, CACR, and AECF implementations, the full ablation suite, formal configurations, experiment commands, and comprehensive tests. It is distributed to authorized reviewers through a separate controlled channel. See `REVIEWER_PACKAGE_MANIFEST.md` for the expected package contents.

## Reproducibility Statement

This public preprint repository is not sufficient to reproduce the paper's complete experimental results. It documents the architecture, source mapping, stable interfaces, configuration contract, and non-confidential infrastructure. Authorized reviewers can verify the complete implementation and experiments with the confidential package and its integrity manifest.

## Citation

Please cite the SACER-RAG preprint and the versioned repository release. Use the bibliographic metadata published with the corresponding preprint record; this repository does not invent or duplicate unpublished citation metadata.
