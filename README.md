[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/t6HJW9qy)
# Module 3 Exam - Brazil Weather Station Observational Data
This repository template is used to perform a comparative analysis between different types of Time Series Models, using traditional Time Series Analysis, and Deep Learning.

This repository is a requirement in part of the Module 3 Exam in DS150 - Time Series Analysis and Stochastic Processes.

### Dataset details
The `sudeste` dataset can be accessed through the following [link](https://dsstorageplayground.blob.core.windows.net/container-data/sudeste.csv). The following data contains observational weather metrics such as temperature.

For details on the measurement data and column names in the file, the following table is given:

<details><summary>Data Dictionary</summary>

| Column Name | Description |
| ---- | ------------------ |
| wsid | Weather Station ID |
| wsnm | Name station (usually city location or nickname) |
| elvt | Elevation |
| lat| Latitude |
| lon | Longitude |
| inme | Station number (INMET number) for the location |
| city | City |
| prov | State (Province)|
| mdct | Observation Datetime (complete date: date + time)|
| date | Date of Observation |
| yr | Year |
| mo | Month |
| day | Day of Month |
| hr | Hour |
| prcp | Amount of precipitation in millimetres (last hour) |
| stp | Air pressure for the hour in hPa to tenths (instant) |
| smax | Maximum air pressure for the last hour in hPa to tenths|
| smin | Minimum air pressure for the last hour in hPa to tenths |
| gbrd | Solar radiation KJ/m2 |
| temp | Air temperature (instant) in celsius degrees |
| dewp | Dew point temperature (instant) in celsius degrees |
| tmax | Maximum temperature for the last hour in celsius degrees |
| dmax | Maximum dew point temperature for the last hour in celsius degrees |
| tmin | Minimum temperature for the last hour in celsius degrees |
| dmin | Minimum dew point temperature for the last hour in celsius degrees |
| hmdy | Relative humid in % (instant) |
| hmax | Maximum relative humid temperature for the last hour in % |
| hmin | Minimum relative humid temperature for the last hour in % |
| wdsp | Wind speed in metres per second |
| wdct | Wind direction in radius degrees (0-360) |
| gust | Gust speed in metres per second |

</details>

### Instructions
In the `src/` folder, modify the given notebook to perform your comparative analysis. You may also include additional `.py` files in this folder for your development if necessary.

Additionally, provide a brief overview of the architecture, a guide in reproducing the given results, and the results itself of your deep learning model when creating your pull request. Additionally, include the steps and methods in creating your TSA model.

### Expected Results
A working `tensorflow` and traditional time series `statsmodels` (or any other applicable libraries) showing results comparison, and implications on the performance between two models.

### Requirements
A merged pull request as a team containing the model framework with successful revisions and status checks.

### Grading Guidelines
This Module 2 Exam would be graded on the following criteria:
- Notebook analysis content and model performance
- Code quality and code workspace efficiency
- Version control mastery
- Adaptation to revisions and changes
