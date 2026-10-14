# Data

`hour.csv` is the hourly file of the **Bike Sharing Dataset** (UCI Machine
Learning Repository, id 275), unchanged from the original download:

- Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Original data: Capital Bikeshare (Washington DC) trip logs for 2011-2012,
  aggregated per hour by the dataset authors and joined with weather and
  calendar information.
- 17,379 rows x 17 columns, 1.1 MB. One row = one hour in which at least one
  ride happened (hours with zero rides are absent, see REPORT.md).
- License/citation: Fanaee-T, H. and Gama, J., "Event labeling combining
  ensemble detectors and background knowledge", Progress in Artificial
  Intelligence (2013). The authors' original `Readme.txt` is kept alongside as
  `UCI_Readme.txt`.

Column notes that matter for this project (full list in `UCI_Readme.txt`):

| column | meaning |
|---|---|
| `dteday`, `hr` | date and hour of day (0-23) |
| `yr`, `mnth`, `weekday`, `holiday`, `workingday` | calendar; `workingday` = 1 when neither weekend nor holiday |
| `weathersit` | 1 clear, 2 mist/cloudy, 3 light rain/snow, 4 heavy rain/ice (3 rows only) |
| `temp`, `atemp`, `hum`, `windspeed` | normalised weather: temp/41 C, feels-like/50 C, humidity/100, wind/67 |
| `casual`, `registered`, `cnt` | rides by user type and in total; `cnt` is the target |

Nothing in `data/` is modified by the pipeline; all cleaning happens in
`pipeline.py` at run time.
