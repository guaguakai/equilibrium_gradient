This is the implementation of "Coordinating Followers to Reach Better Equilibria: End-to-End Gradient Descent for Stackelberg Games" submitted to NeurIPS 2021 with paper ID: 3668. In this folder, you can find three different subfolders, `qre`, `ssg`, and `cyber`, which refer to three examples presented in the paper.

All methods are implemented in Python3.6, where a list of the dependency can be found in `package-list.txt`.

`model.py` contains the implementation of an iterative Nash equilibrium oracle and a pytorch implementation of the stochastic differentiable Nash equilibria layer. Within each folder, `main.py` is the major implementation of the corresponding domain. You can simply run `python main.py --method=$METHOD` with various args to run the experiments.
For reproducibility, you can specify the random seed for running experiments.

Our method is called `diffmulti`. Additionally, there are three different baselines: `SLSQP`, `trust`, and `knitro`, where the first two use `scipy.optimize.minimize` to run blackbox optimization, and the last one is the implementation of variational inequality reformulation using knitro solver. You can obtain a trial license from [Knitro website](https://www.artelys.com/solvers/knitro/) to run the experiment. 
