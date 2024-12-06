# Semantic Segmentation

## Structure
-  `models`: Contains the semantic segmentation model definitions.
-  `outputs`: Contains the trained models.
-  `src`: Source code, containing metrics, loss functionsm loaders, etc.
-  `train_model.py`: Training script.
-   `test_model.py`: Testing script.
  
## Usage
**Training**:
1. Ensure that images are divided into patches and correctly preprocessed (see preprocessing directory)
2. Select training configuration inside the `train_model.py` script.
3. Run script (this will save the best model in the outputs directory).

**Testing**:
1. Ensure that images are divided into patches and correctly preprocessed (see preprocessing directory)
2. Select test configuration inside the `test_model.py` script.
3. Run script (this will print the confusion matrix and metrics).