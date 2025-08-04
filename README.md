# SGF-BLO
Safe Bilevel Optimization

This repository contains the codebases for the methods described in [**"Safe Gradient Flow for Bilevel Optimization"**](https://arxiv.org/abs/2501.16520), presented at the **2025 American Control Conference (ACC)**, [**"Sequential QCQP for Bilevel Optimization with Line Search"**](https://arxiv.org/abs/2505.14647?) accepted at **IEEE Control Systems Letters (L-CSS)** and **2025 Conference on Decision and Control (CDC)**, and [**"Perturbed gradient descent via convex quadratic approximation for nonconvex bilevel optimization"**](https://arxiv.org/abs/2504.17215) which is under review.

## How to Run the Code

To run the code, use the following command: `bash run.sh`

This command runs the following code: ` python3 main.py --testID x --senarioID y`.
You can modify the configurations based on your experiment.

Here, `main.py` is the main script, and `--scenarioID` chooses the method to compare with and the parameters for the experiment. The default setup is the DHC, and by adding a flag `--toy_example`, the benchmark can be modified.


## Citation

If you use this code in your work, please cite our papers:

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