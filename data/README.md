# Data provenance

- Dataset: **Bank Marketing**, UCI Machine Learning Repository.
- Creators: Sérgio Moro, Paulo Rita and Paulo Cortez.
- Dataset DOI: https://doi.org/10.24432/C5K306
- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Download: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Retrieved: 2026-10-06. The original `bank-additional.zip` inside the download contains both files retained here.
- License: **CC BY 4.0**, as stated by UCI. https://creativecommons.org/licenses/by/4.0/
- File: `bank-additional-full.csv`, semicolon-delimited, 41,188 rows, 20 inputs plus `y`, 5,834,924 bytes.
- Original order retained; the supplied documentation says records are ordered by date, May 2008-November 2010. Exact dates and customer IDs are absent.
- `bank-additional-names.txt` is the original data dictionary, retained unchanged.
- No dataset edits or row filtering. The SHA-256 is recorded by training in `artifacts/metadata.json`.
- This version uses **999**, not -1, as the `pdays` no-previous-contact sentinel. UCI's general webpage also describes an older variant; the included dictionary governs this file.

Publication: S. Moro, P. Cortez and P. Rita (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22-31. https://doi.org/10.1016/j.dss.2014.03.001

Observed dictionary/data inconsistency: 4,110 rows use pdays=999 while previous>0. We preserve both fields and describe the indicator as recency availability, not conclusive absence of past contacts.
