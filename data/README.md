# Dataset Overview
This folder contains `AI_Classification-Project.csv` (6,557 rows), which holds metadata and image URLs for the Authentic Face Classification project.

## WARNING: DATA LEAKAGE COLUMNS
Do not use the following columns as model features, as they map 1:1 with the target label and will cause a false 100% accuracy:
* `detection_difficulty`
* `category`
* `label_numeric`
* `source`
* `fake_method`