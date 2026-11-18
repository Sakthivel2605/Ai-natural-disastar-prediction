# AI Natural Disaster Prediction

Predicts flood risk from rainfall, river level, and soil moisture with logistic regression.

> **Status:** concept prototype. It runs on simulated data and has not been deployed or tested with real hardware or real-world data.

## What it is
A Python machine-learning example. It generates a simulated dataset, trains a logistic-regression model, reports accuracy, precision, and recall on held-out data, and gives a flood risk percentage for any set of readings.

## Why I made it
Early warning gives people time to move to safety. I wanted a small, honest first model for how weather data can feed a warning system, building on my Weather Forecasting System.

## Who it is for
Disaster-management students and hobbyists learning applied machine learning. It is not for real emergency decisions.

## How it was made
1. 3,000 simulated days are generated with rainfall (mm in 24 h), river level (m), and soil moisture (%).
2. A made-up rule turns these into a flood probability, and labels are drawn from it. This is the simulation's assumption, not real hydrology.
3. A scikit-learn pipeline (StandardScaler then LogisticRegression) is trained on 80% and tested on 20%.
4. `predict_proba` gives a risk percentage, shown as LOW, MEDIUM, or HIGH.

## Languages and tools
Python 3, NumPy, scikit-learn. Tests with pytest.

## Run it
```bash
pip install -r requirements.txt
python flood_predictor.py
python flood_predictor.py --rain 10 --river 2.5 --soil 30
```

Example output:
```text
Test metrics on simulated data: {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.8}
Flood risk for rain=120 mm, river=4.2 m, soil=80%: 100% (HIGH)
```

Run the tests with `pip install pytest` and then `pytest`.

## Limitations
The model only learns the rule used to generate the data, so its scores say nothing about real floods. Real prediction needs historical records and local expert validation. Do not use it for safety decisions.

## Future work
Train on public data (rainfall and river-gauge records), compare Random Forest and gradient boosting, add other hazards, and connect weather APIs.

## Author
Sakthivel. Computer Science and Engineering, Lovely Professional University.

## License
MIT. See [LICENSE](LICENSE).
