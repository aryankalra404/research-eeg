# September 2026 documents

- `STEW_Results.docx`: compact results sheet (tables, figures, short notes; no cover page or names). Built by `results.js`.
- `STEW_Results.pdf`: PDF of the same, made by `topdf.py` (docx → HTML via mammoth → Chromium print).
- `STEW_Progress_Report.docx`: full narrative version with title page. Built by `build.js`.

Regenerate:
```bash
python3 figs.py .            # figures from docs/RESULTS.md (run from the repo root with PYTHONPATH=src)
npm install docx && node results.js && node build.js
```
Numbers are copied from `docs/RESULTS.md`. ACCT (Model E) values are development
(validation-subject) results, not the final test.
