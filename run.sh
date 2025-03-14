#!/bin/bash
# --------------------------------
# testID: 0 -> toy_example, 1 -> toy_example nonconvex 
#           2 -> toy_CS, 3 -> DHC with PCA 4 -> DHC Large scale
#          5 -> DHC with NN
# --------------------------------
# scenarioID: Choose from setup.py to run the experiments of the ACC 2025 or ICML 2025 paper

python3 main.py --testID 0 --scenarioID 10

