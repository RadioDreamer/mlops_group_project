# Evidently Reports

Evidently is used to generate data and model monitoring reports. The project stores generated reports in `reports/evidently_reports/`.

## Generating reports

- See `src/fakeartdetector/evidently_report.py` for how reports are created.
- Typical flow:
  - Collect recent inference logs or dataset slices
  - Run the report script to produce interactive HTML reports

## Viewing reports

- HTML reports are stored in `reports/evidently_reports/` and can be opened in a browser.

## Use cases

- Model drift detection
- Data quality checks
- Performance comparisons between models
