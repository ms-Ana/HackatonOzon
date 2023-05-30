## Overview of solution
```
.
├── data_utils
│   ├── colors.txt
│   ├── Remove_errors.ipynb
│   ├── test_colors.txt
│   └── Transitivity_preprocessing.ipynb
├── Main.ipynb
├── README.md
├── requirements.txt
└── utils
    ├── cat_features.py
    ├── image_features.py
    ├── metric.py
    ├── text_features.py
    └── training.py
```
 ### Data preprocessing
 The folder data_utils contains all functions need to prepare data for training.
 1. Transitivity_preprocessing - contains expansion data with transitivity, and errors removing
 2. Remove_errors - contains parsing categories, removing errors in categories, color preprocessing
 ### Utility functions
 The folder utils contains all functions to create features for training
 1. cat_features - parsing characteristic_attributes_mapping
 2. image_features - distances for image embeddings
 3. metric - sugested metric for quality measurement
 4. text_features - functions to create text data features
 5. training - train utility functions
 ### Main
 Main.ipynb contains improved baseline