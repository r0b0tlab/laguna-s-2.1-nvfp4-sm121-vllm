# Laguna S 2.1 NVFP4 DFlash K=7 — Full 8,620-Case Scorecard

Status: **PASS**
Suite: `r0b0bench-core-v1-rc2`
Immutable image: `sha256:8b0e3d07dad370853bca77441ff7c8619c41a0cb80d59a150bb0faf17ee15ef3`
Quality concurrency: **c2**

## Official BFCL
- BFCL v4 `multi_turn_base`: **137/200 (68.50%)**

## Generated-answer tasks

| Test | Correct / N | Accuracy | Extraction failures | Length stops |
|---|---:|---:|---:|---:|
| arc_challenge | 1122 / 1172 | 95.73% | 0 | 0 |
| gsm8k | 1260 / 1319 | 95.53% | 0 | 0 |
| mmlu_abstract_algebra | 81 / 100 | 81.00% | 11 | 0 |
| mmlu_business_ethics | 82 / 100 | 82.00% | 0 | 0 |
| mmlu_clinical_knowledge | 229 / 265 | 86.42% | 0 | 0 |
| mmlu_college_biology | 139 / 144 | 96.53% | 0 | 0 |
| mmlu_computer_security | 83 / 100 | 83.00% | 0 | 0 |
| mmlu_conceptual_physics | 219 / 235 | 93.19% | 0 | 0 |
| mmlu_high_school_world_history | 215 / 237 | 90.72% | 0 | 0 |
| mmlu_international_law | 109 / 121 | 90.08% | 0 | 0 |
| piqa | 1710 / 1838 | 93.04% | 0 | 0 |
| truthfulqa_mc1 | 583 / 817 | 71.36% | 0 | 0 |
| winogrande | 1006 / 1267 | 79.40% | 0 | 0 |

Generated weighted lane: **6838/7715 (88.63%)**

## IFEval
- Prompt strict: **82.44%**
- Prompt loose: **86.69%**
- Instruction strict: **87.53%**
- Instruction loose: **90.89%**
- Length stops: 2/541

## HumanEval
- pass@1: **154/164 (93.90%)**
- Sandbox infrastructure: **PASS**

## Method and diagnostics
- Official logical cases: 8,620/8,620
- Excluded calibration: 210/210; semantic disagreements: 6
- Generated extraction failures: 11
- Generated empty outputs: 0
- Generated length stops: 0
- There is no opaque composite score. Each pillar is reported separately.

## Reused qualified performance/context evidence
- Source SHA-256: `43600a788229ba27ce8abf0f39c6303ff49d4752bc348e94cb0786bcfa3f7f60`
- Performance: PASS
- Context: PASS
- Nsight/native gate: PASS
