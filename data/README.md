# Implemented Datasets

There are three currently implemented datasets:

## FORCE Dataset

### Overview
This dataset was taken from the FORCE 2020 competition.

### Dataset Access
To access the dataset, CSV files for the following subsets need to be opened:

- **Train Data**: This is the data used to train models.
- **Leaderboard Test Data**: Data provided for real-time evaluation during the competition.
- **Hidden Test Data**: Released post-competition for final evaluation.

### Competition Details
During the competition, the test data was split into two parts:
1. **Leaderboard Test Data**: Available to all participants during the competition for the purpose of provisional ranking.
2. **Hidden Test Data**: Released only after the competition ended, used for generating the final leaderboard.

## Taranaki Dataset

### Overview
This dataset comes from the Taranaki basin in New Zealand.

### Dataset Access
The dataset is comprised of a single csv file, with all logs and classes from all wells. Accessing this information is pretty straightforward. It involves a single-step process:

1. **File Opening**: Use the `pandas` library to read the CSV file.
