# Datasets

## 1. Daily Delhi Climate

Daily weather observations for Delhi, India (source: Weather Underground API; published on Kaggle as
"Daily Climate time series data" by Sumanth Rao).

| File | Period | Days |
|---|---|---|
| `DailyDelhiClimateTrain.csv` | 2013-01-01 to 2017-01-01 | 1,462 |
| `DailyDelhiClimateTest.csv` | 2017-01-01 to 2017-04-24 | 114 |

| Feature | Unit |
|---|---|
| meantemp | °C (mean of 3-hourly readings) |
| humidity | % |
| wind_speed | km/h |
| meanpressure | hPa |

Pre-processing in `Code/data.py`:
- `meanpressure` contains sensor glitches (e.g. 59 hPa, 7,679 hPa); values outside 950–1050 hPa are treated as missing
  and linearly interpolated.
- The training file's last row (2017-01-01) duplicates the first test day and is dropped.
- Min-max scaling fitted on the training period only; 7-day input windows → next-day prediction of all 4 variables.

## 2. Crop Yield Prediction (`yield_df.csv`)
Kaggle "Crop Yield Prediction Dataset" (Patel), compiled from FAO (yield, pesticides) and World Bank (rainfall,
temperature) data: 28,242 rows, 101 countries, 10 crops, 1990–2013.

| Column | Description |
|---|---|
| Area | country |
| Item | crop (maize, potatoes, rice, wheat, ...) |
| Year | 1990–2013 |
| hg/ha_yield | yield, hectograms per hectare (**target**) |
| average_rain_fall_mm_per_year | mm |
| pesticides_tonnes | tonnes of active ingredient |
| avg_temp | °C |
