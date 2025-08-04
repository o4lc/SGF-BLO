#!/bin/bash
# --------------------------------
# testID: 0 -> toy_example, 1 -> toy_example nonconvex, 2 -> toy_example constrained 
#           3 -> toy_CS, 4 -> DHC with PCA 5 -> DHC Large scale
#          6 -> DHC with NN
# --------------------------------
# scenarioID: Choose from setup.py to run the experiments of the ACC 2025 or CDC 2025 paper

python3 main.py --testID 2 --scenarioID 10

