# SGF-BLO
Safe Gradient Flow for Bilevel Optimization

This codebase implements the methods described in the paper **"Safe Gradient Flow for Bilevel Optimization"**, accepted at the **2025 American Control Conference (ACC)**.
You can find the full paper on [arXiv here](https://arxiv.org/abs/2501.16520) 

## How to Run the Code

To run the code, follow these steps:
To train a model, use the following command: `bash run.sh`
This command runs the following code: ` python3 main.py --senarioID x `
Here, `main.py` is the main script, and `--scenarioID` chooses the method to compare with and the parameters for the experiment. The default setup is the DHC, and by adding a flag `--toy_example`, the benchmark can be modified.
You can modify the configurations based on your experiment.

## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{sharifi2025safe,
  title={Safe Gradient Flow for Bilevel Optimization},
  author={Sharifi, Sina and Abolfazli, Nazanin and Hamedani, Erfan Yazdandoost and Fazlyab, Mahyar},
  journal={arXiv preprint arXiv:2501.16520},
  year={2025}
}

