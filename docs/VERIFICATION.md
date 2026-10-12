# Verification record

Verified locally on 6 October 2026. The exact dependency versions, data hash, split membership, selected settings and numerical agreement are stored in `artifacts/`.

## End-to-end execution

- `uv sync --frozen`: completed against the provided lockfile (112 installed packages).
- `uv run python main.py train`: completed using the bundled local CSV, no prompts or download.
- Manual and standard PyTorch each converged after 6,433 updates at gradient infinity norm below 1e-7.
- All three models made identical test decisions. Maximum probability difference from scikit-learn: 0.00000560805.
- `uv run python -m pytest -q`: **11 passed** in the final source checkout.
- An earlier extracted backup was retrained from an empty artifacts directory, invoked from a different working directory; it reproduced the numerical results and passed its then-current 10 checks. The final schema rename preserves feature values and is covered by the final 11-check run.
- `uv run python main.py app`: served HTTP on `127.0.0.1:7860` using saved files.
- `uv run python scripts/verify_app.py`: **31 API checks passed**, covering all model options, split comparisons, threshold endpoints, feature/target plots, saved-model record inference and reset. All artifact hashes remained unchanged.

## Browser and presentation checks

- Verified validation default cost EUR 16,444 and test default cost EUR 17,882.
- Moved the browser threshold slider to 1: test cost EUR 127,000 (call nobody).
- Moved it to 0: test cost EUR 16,476 (call everybody).
- Restored the frozen policy and verified EUR 17,882 again.
- Inspected model-comparison table, target-distribution chart and saved-model inference controls.
- Rendered and visually reviewed the 13-page PDF, including table wrapping, page starts, footers and final numerical consistency.

## Issues caught and resolved

- Starter provided both .py and .ipynb for each stage: moved unused starter notebooks under docs/, leaving one root entrypoint per stage. main.py is unchanged.
- Exact score-boundary threshold could change a decision after floating-point serialization: use a validation-score midpoint and round-trip CSV parsing. A test compares all displayed policy counts/costs against exported metrics.
- pdays=999 coexists with previous>0 on 4,110 rows: preserve both signals, name the indicator recency_recorded, and document the dictionary/data inconsistency. The renaming does not change numerical feature values.

## Scope of evidence

These are local macOS checks with the locked environment. The supplied GitHub Actions workflow is retained, but no remote CI execution is claimed. No fork push, upstream pull request, Blackboard upload or submission was performed. The poor test policy-cost result is retained rather than hidden or retuned on test.

## Final archive check

The final packaged source and artifacts were extracted to a separate temporary directory. Its app was launched through its own `main.py app` from `/tmp` on port 7861, responded with HTTP 200, and passed all 31 API checks without modifying artifact hashes. That temporary server was stopped; the main dashboard remains available on port 7860.


## Dashboard clarity redesign

The dashboard now opens on the frozen test result, with cost arithmetic and a stacked cost comparison. Policy controls use percentages and include explicit call-all, call-none and restore buttons. Confusion-matrix cells name the outcomes, model metrics have visual comparisons and definitions, target distributions display percentages, and technical details expand on demand.

Validation after this UI change: all 11 existing tests passed; 37 Gradio API checks passed with all saved-artifact hashes unchanged. Browser checks confirmed the new endpoint buttons, test/validation selection and restore behavior. Overview and model-comparison screens were visually inspected. The PDF demonstration page and backup ZIP were updated. No model retraining, publishing or submission occurred.

## Final prose and editorial dashboard verification — 12 October 2026

This section supersedes the earlier publication-status statements. Walter revisions were checked against the measured evidence, with factual errors corrected and the student's interpretation preserved. The guide was rebuilt and rendered; all 13 pages were reviewed, including correction of an orphaned final line after prose expansion.

The student requested an editorial dashboard with a professional banking-product feel. The overview now pairs its cost conclusion with a large comparison chart, followed by the arithmetic and key context. The policy workspace separates controls from results, and secondary assumptions expand on demand. All five grading views remain available.

- `uv run python -m pytest -q`: **11 passed** after the prose and dashboard edits.
- `uv run python scripts/verify_app.py`: **37 API checks passed**, with every saved artifact hash unchanged.
- Browser verification: final navigation, overview, model charts, test call-none cost EUR 127,000, and restored saved-policy cost EUR 17,882.
- Visual review: original-color in-app preview and Chrome preview; corrected a clipped chart total at a narrower viewport. Chrome's Dark Reader extension alters the displayed palette without changing the app.
- `git diff --check`: no whitespace errors.
- See `DELIVERY.md` for the subsequent fork/PR and remote workflow status. Blackboard remains outside the authorized actions.
