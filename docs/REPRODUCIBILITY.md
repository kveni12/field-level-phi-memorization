# Artifact inventory and reproducibility limits

This public release was selected from the uncommented manuscript supplied on November 19, 2026. Its scope is the synthetic matched corpus, SFT, prefix attacks, first-token J-Lens and the onset statistics that remain in the prose. The onset figure itself is commented out in that manuscript and is not released as a paper figure. Autoregressive generation, template/oracle concepts and regenerated SSN data are outside this release. A subsequent author-requested Figure 3 update uses SSN-last-four token mean RR on the reconstructed step-800 model, matching the new Figure 4; the other field metrics are unchanged. This update is separate from the archived fourteen-field statistics cited in the original prose.

## What is reproducible here

1. The 2,000 synthetic document rows reproduce exactly from frozen pools; paired split membership is verified.
2. The prefix heatmap and historical first-token-only J-Lens heatmap reproduce from archived cell colors. Current Figure 3 instead uses numerical means in `results/current/jlens_heatmap_inputs.json`, with all plotted onset crossings verified against the fresh Figure 4 analysis. Historical colors are not numerical observations.
3. The appendix aggregate panel is preserved as an archived PNG. Raw per-example attack observations for that historical panel are not included.
4. SFT, prefix attack, Jacobian fitting, first-token scoring, numerical heatmap plotting and paired-bootstrap onset analysis can run on newly trained checkpoints. CPU tests check the core operations. No fresh 8B GPU run is claimed as validation of this release.
5. `results/archived/onset_statistics.json` preserves the original reported fourteen-field statistics. `analyze_onsets.py` computes the same first-crossing estimator from new raw score rows and requires an explicit checkpoint step rather than hard-coding one.

## Items that must not be silently conflated

| Manuscript/artifact | Retained implementation or available evidence |
|---|---|
| “50% of tokens” prefix | Current code uses about 50% of characters, aligned to a decoded token boundary. Historical exact-half-token code would change eligibility. |
| J-Lens figure caption says step 800 | Archived onset metadata says step 600. Corresponding historical checkpoint/score provenance remains unresolved. |
| Mean onsets 23.00 / 24.14; fourteen-field significance | These numbers summarize the archived full fourteen-field analysis, while the displayed heatmap contains eleven selected rows. |
| Phone/hotline context discussion | Both twins share their phone digits and geography; hotline is not independent of the earlier geographic fields. |
| Fourteen identical shared identifiers | The actual corpus has thirteen identical structured values; names and initials/code differ between twins. City, state and ZIP are separate fields. |
| Maximum retention difference “3.25%” | The historical raw observation file needed to independently verify this number was not recovered. An absolute rate difference should be distinguished from a relative percent change. |
| Overview `fig_1.png` | The manuscript illustration was not present in the available repository; analysis code and the three other active figure assets are included. |

These are recorded discrepancies, not modifications to the paper's outcomes. Do not substitute a fresh checkpoint's scores or invert a colormap to manufacture missing historical numerical data.

## Data and estimator conventions

The prefix evaluator normalizes case/whitespace and requires a complete value in its field context. It scores only complete fields eligible after the split, with eligible-instance denominators. Fields already supplied in the prefix are not counted as successful extraction targets.

J-Lens uses gold preceding document text and one first data-token target for each field. The default skips MRN-/HPB- literals but retains `000`/`999` for SSN/account. The current Figure 3 uses `--ssn-last-four`: it supplies the SSN prefix and averages RR over every final-four-digit token. This row therefore has a different target span than the other rows and is labeled accordingly. Source layers use fitted Jacobian maps; the final endpoint uses the retained readout convention. Fitted caches must have a matching checkpoint/tokenizer manifest.

Onset is the first layer where cohort mean reciprocal rank reaches 0.2. It need not remain above threshold in later layers. A transient early crossing can therefore yield a large apparent gap. Paired bootstrap resampling uses matched-pair IDs; the original comparison family has fourteen fields. Neither onset nor MRR establishes a causal memory location or full-value extraction.

## Files and provenance

`RELEASE_SOURCES.json` records source-file hashes and retained helper roots for the code cleanup. `FILE_MANIFEST.json` fingerprints the published release files. The new public repository starts with a clean history and does not include the private working repository's experiments, logs, credentials, manuscript drafts or cluster paths.

The archived paper heatmap metadata is deliberately kept as historical evidence, including its conflicting step label. Runnable entry points accept local paths rather than embedding the original cluster's home or scratch directories.
