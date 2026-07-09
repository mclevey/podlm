# `transformer_opinion_dynamics` Lineage

Date: 2026-07-09
Source repository: `/Users/johnmclevey/development/transformer_opinion_dynamics`
Active repository: `/Users/johnmclevey/development/podlm`

## Decision

`transformer_opinion_dynamics` is archived provenance. `podlm` is the active
home for the SSHRC opinion-dynamics research software.

Do not port large surfaces from `odlm` into `podlm`. The old repository is
useful for lineage, a few small ideas, and bug discovery. Its architecture,
environment, and security posture should not be revived.

## Repository Lineage

The archived repository was a project-specific `pdpp` pipeline and helper
package named `odlm` for Opinion Dynamics with Language Models. It collected
Reddit submissions and comments from an Elasticsearch mirror, sampled
discussion data, segmented posts into sentences, labeled text with transformer
models, created topic models, and prepared features for opinion-dynamics
analysis.

`podlm` supersedes it:

| Archived surface | Active `podlm` surface | Status |
|---|---|---|
| `odlm/odlm/reddit.py` | `podlm/podlm/reddit.py` | Superseded. `podlm` uses injected Elasticsearch configuration instead of a module-level credential. |
| `odlm/odlm/text.py` | `podlm/podlm/text.py` | Superseded, with retained labeling and topic-model ideas. |
| `dodo.py` and task dirs | `pipeline/` | Superseded by the larger `podlm` pipeline. |
| root dependency PNGs | `docs/lineage/transformer_opinion_dynamics/*.png` | Preserved as provenance. |
| empty network/similarity stubs | `podlm/podlm/networks.py`, `pipeline/compute_ij_similarities`, `pipeline/blockmodel_networks` | Implemented in `podlm`. |
| SetFit notebooks | `pipeline/label_with_few_shot_learners`, future recipes | Conceptually migrated. |

## Fragments Ported Forward

This pass intentionally ported only small fragments that improved `podlm`
without reviving the old package:

1. Archive lineage documentation in both repositories.
2. Historical dependency graph images under
   `docs/lineage/transformer_opinion_dynamics/`.
3. Optional BERTopic POS representation support from the old modular topic
   pipeline, while keeping it disabled by default.
4. Text-labeling correctness fixes found by auditing the old lineage:
   sentence alignment across non-string rows, tokenizer-level truncation when
   available, no raw-text error logging, fail-loud model errors, and
   row-aligned emotion-label outputs.
5. Entity count merging via `entities_info` rather than the long entity table.

## Risks Not Ported

Do not carry these old patterns forward:

- module-level Elasticsearch clients;
- committed credentials or endpoint strings;
- `verify_certs=False` defaults;
- raw user text in error logs;
- bare `except:` blocks in analytical labeling code;
- position-based zips that can silently misalign model outputs;
- hard-coded `../input` and `../output` paths in new public APIs;
- Python 3.7-era environment pins.

## Historical Graphs

These files preserve the archived `pdpp` graph as it existed in
`transformer_opinion_dynamics`:

- `transformer_opinion_dynamics/dependencies_all.png`
- `transformer_opinion_dynamics/dependencies_file.png`
- `transformer_opinion_dynamics/dependencies_source.png`
- `transformer_opinion_dynamics/dependencies_sparse.png`

They are provenance artifacts. Do not treat them as the active `podlm` task
graph.

## Workbench Evidence

The canonical audit is filed in the workbench:

```text
/Users/johnmclevey/workbench/reports/2026-07-09-report-transformer-opinion-dynamics-audit-roadmap-fable5.md
```

Follow-on `podlm` work should start with tests and contracts before method
expansion.
