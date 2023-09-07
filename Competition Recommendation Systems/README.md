# Method Description

To improve the accuracy and efficiency of this recommendation system, several features have been introduced. The decision on which feature to add to the XGBoost model was made based on the calculation of their importance. Features with higher importance were prioritized for this model. 

To prevent overfitting:
- The training dataset was divided into 5 groups.
- In each iteration, the model was trained on 4 groups and tested on the remaining group.
- This process was repeated 5 times to ensure accuracy.
- Stacking was utilized as a final step to boost accuracy.

## Error Distribution:

- **>=0 and <1:** 102250
- **>=1 and <2:** 32835
- **>=2 and <3:** 6149
- **>=3 and <4:** 810
- **>=4:** 0

## RMSE:
0.9792574100552962

## Execution Time:
982 seconds

