# SGF-BLO
Safe Bilevel Optimization: In these series of works, we take inspiration from control theorey to design methods for solving bilevel optimization problems.

This repository contains the codebases for the methods described in [**"Safe Gradient Flow for Bilevel Optimization"**](https://arxiv.org/abs/2501.16520), presented at the **2025 American Control Conference (ACC)**, [**"Sequential QCQP for Bilevel Optimization with Line Search"**](https://arxiv.org/abs/2505.14647?) accepted at **IEEE Control Systems Letters (L-CSS)** and **2025 Conference on Decision and Control (CDC)**, and [**"Perturbed gradient descent via convex quadratic approximation for nonconvex bilevel optimization"**](https://arxiv.org/abs/2504.17215) which is under review.

Next, we discuss how to execute the code.

## How to Run the Code
To run the code, use the following command: `bash run.sh`.

This command runs the following code: ` python3 main.py --testID x --senarioID y`.
Here, `main.py` is the main script, `--testID` selects the experiment setup from different options, and `--scenarioID` chooses the method to compare with and the parameters for the experiment. 

You can perform the experiments with all the developed methods:
- `IFCT` denotes the continuous-time method developed in "Safe Gradient Flow for Bilevel Optimization".
- `IFDT` denotes the discrete-time method developed "Sequential QCQP for Bilevel Optimization with Line Search" and "Perturbed gradient descent via convex quadratic approximation for nonconvex bilevel optimization". 
  - You can then set the mode from `[QP1, QP2, QCQP]` to specify which method you want to call. 
- `SecondOrder` denotes the prediction-correction method discussed in the appendix of "Safe Gradient Flow for Bilevel Optimization".

You can also choose from the following problems:
- `toy_example` and `toy_example_nc` denote the convex and nonconvex synthetic examples, respectively.
- `toy_CS` denotes the small-scale coreset selection problem.
- `DHC` denotes the small scale data hyper cleaning (DHC) problem, and `DHC_LS` denotes the large scale DHC.
- `NN` denotes the DHC problem with a neural network classifier.

The `BilevelSolver` class in `bilevel_solver.py` containts the implementations of our methods, and the state-of-the-art methods that we comapre with. You can also find the code for calculating the function definitions for `f` and `g`, and their derivatives in the same file. 

You can modify the configurations based on your experiment. To add experiment with new parameters or method, you can change the `scenario_setup()` function in `setup.py`.

## Citation
If you use this code in your work or found this repository usefull, please cite our papers:

```bibtex
@article{sharifi2025safe,
  title={Safe Gradient Flow for Bilevel Optimization},
  author={Sharifi, Sina and Abolfazli, Nazanin and Hamedani, Erfan Yazdandoost and Fazlyab, Mahyar},
  journal={arXiv preprint arXiv:2501.16520},
  year={2025}
}

@article{abolfazli2025perturbed,
  title={Perturbed gradient descent via convex quadratic approximation for nonconvex bilevel optimization},
  author={Abolfazli, Nazanin and Sharifi, Sina and Fazlyab, Mahyar and Hamedani, Erfan Yazdandoost},
  journal={arXiv preprint arXiv:2504.17215},
  year={2025}
}

@article{sharifi2025sequential,
  title={Sequential QCQP for Bilevel Optimization with Line Search},
  author={Sharifi, Sina and Hamedani, Erfan Yazdandoost and Fazlyab, Mahyar},
  journal={arXiv preprint arXiv:2505.14647},
  year={2025}
}
```