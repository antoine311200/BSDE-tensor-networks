
<h2 align="center">PDE Solver using Tensor Trains</h2>

![version](https://img.shields.io/badge/version-0.0.1-blueviolet)
![development](https://img.shields.io/badge/development-in%20progress-orange)
![maintenance](https://img.shields.io/badge/maintained-yes-brightgreen.svg)
![launched](https://img.shields.io/badge/launched-no-red.svg)


# Introduction

This project stems from a last year research project at CentraleSupélec supervised by the Crédit Agricole CIB. The goal of the project was to solve Partial Differential Equations (PDEs) in high dimensions using Tensor Trains. It reimplements the work done in the paper [Solving high-dimensional parabolic PDEs using the tensor train format](https://arxiv.org/abs/2102.11830) by Lorenz Richter, Leon Sallandt, Nikolas Nüsken (2021) and takes it a step further with complete benchmarking and testing.

In ther original work, the SALSA algorithm was used for the optimisation of the tensor trains with some further modifications by adding the payoff directly to the basis. In accordance with our result and discussion with the Crédit Agricole CIB, we implement our own version of three different ALS algorithm variants:

- The original ALS algorithm for tensor trains
- The Modified Alternating Least Squares (MALS) algorithm (work in progress)
- The Stable Alternating Least Squares Approximation (SALSA) algorithm

Each algorithm presents distinct advantages and disadvantages in terms of speed and accuracy. The primary aim of this project was to implement, benchmark, and compare these three algorithms in relation to their speed and accuracy.

# Run Pipeline

There is two type of pipelines that can be run:

- The explicit discretisation scheme
- The implicit discretisation scheme

The implicit one takes an iterative approach at it time steps to further refine the solution hence resulting in a more accurate solution but at the cost of time. The explicit one takes a direct approach at solving the PDEs and hence is faster but less accurate.

In order to run the pipeline, you need to run the following command:

For the explicit scheme:
```bash
python -m src.pipeline
```

For the implicit scheme:
```bash
python -m src.scripts.pipeline_implicit
```

## Use GPU acceleration

In order to use GPU acceleration, you need to install the CuPy library (https://docs.cupy.dev/en/stable/install.html)
Then, in the file `/src/bsde_solver/__init__.py` you need to change the following line:

```python
reload_backend('numpy')
```

to

```python
reload_backend('cupy')
```

# Run Experiments

Various experiments were conducted to benchmark the algorithms.

To run the experiments, you need to run the following command:

```bash
python -m src.scripts.experiments.[experiment_name]
```

for instance, to run the `algo_comparison` experiment, you need to run the following command:

```bash
python -m src.scripts.experiments.algo_comparison
```

The following experiments are available:

- `algo_comparison`: Compares the ALS to the SALSA algorithms in a grid search manner that can be modified (by default the grid search is on the batch size with all other parameters fixed)
- `batch_size`: Compares the effect of the batch size on the ALS algorithm
- `benchmark_cupy`: Compares the performance of the ALS algorithm with and with CuPy instead of NumPy
- `degree_influence` & `degree`: Compares the performance of the algorithms with different degrees of the tensor train
- `implicit_explicit`: Compares the performance of the implicit and explicit schemes
- `n_assets`: Compares the performance of the algorithms with different number of assets
- `n_iter`: Compares the performance of the algorithms with different number of iterations
- `pipeline_comparison`: Compares the performance of the algorithms with the pipelines

# Report

Our report can be found at the root of the repository under the name `Report - Solving_high_dimensional_PDEs_with_tensor_networks - Debouchage - Lemercier.pdf`.

# References

[1] Jacob C Bridgeman and Christopher T Chubb. Hand-waving and interpretive dance: an introductory course on tensor networks. Journal of physics
A: Mathematical and theoretical, 50(22):223001, 2017.

[2] Lukasz Cincio, Jacek Dziarmaga, and Marek M Rams. Multiscale entanglement renormalization ansatz in two dimensions: quantum ising model.
Physical review letters, 100(24):240603, 2008.

[3] Pierre Comon, Xavier Luciani, and Andr´e LF De Almeida. Tensor decompositions, alternating least squares and other tales. Journal of Chemometrics: A Journal of the Chemometrics Society, 23(7-8):393–405, 2009.

[4] Lawrence C Evans. Partial differential equations, volume 19. American
Mathematical Society, 2022.
[5] Mark Fannes, Bruno Nachtergaele, and Reinhard F Werner. Finitely correlated states on quantum spin chains. Communications in mathematical
physics, 144:443–490, 1992.

[6] Lars Grasedyck, Melanie Kluge, and Sebastian Kr¨amer. Alternating least squares tensor completion in the tt-format. arXiv preprint
arXiv:1509.00311, 2015.

[7] Lars Grasedyck and Sebastian Kr¨amer. Stable als approximation in the
tt-format for rank-adaptive tensor completion. Numerische Mathematik,
143(4):855–904, 2019.

[8] Sebastian Holtz, Thorsten Rohwedder, and Reinhold Schneider. The alternating linear scheme for tensor optimization in the tensor train format.
SIAM Journal on Scientific Computing, 34(2):A683–A713, 2012.

[9] Cˆome Hur´e, Huyˆen Pham, and Xavier Warin. Deep backward schemes
for high-dimensional nonlinear pdes. Mathematics of Computation,
89(324):1547–1579, 2020.

[10] Ian P McCulloch. From density-matrix renormalization group to matrix
product states. Journal of Statistical Mechanics: Theory and Experiment,
2007(10):P10014, 2007.

[11] Rom´an Or´us. A practical introduction to tensor networks: Matrix product
states and projected entangled pair states. Annals of physics, 349:117–158,
2014.

[12] Ivan V Oseledets. Tensor-train decomposition. SIAM Journal on Scientific
Computing, 33(5):2295–2317, 2011.

[13] Sebastian Paeckel, Thomas K¨ohler, Andreas Swoboda, Salvatore R Manmana, Ulrich Schollw¨ock, and Claudius Hubig. Time-evolution methods
for matrix-product states. Annals of Physics, 411:167998, 2019.

[14] Roger Penrose et al. Applications of negative dimensional tensors. Combinatorial mathematics and its applications, 1:221–244, 1971.

[15] David Perez-Garcia, Frank Verstraete, Michael M Wolf, and J Ignacio Cirac. Matrix product state representations. arXiv preprint quantph/0608197, 2006.

[16] Nicolas Perkowski. Backward stochastic differential equations: An introduction. Available on semanticscholar. org, 2011.

[17] Maziar Raissi. Forward–backward stochastic neural networks: deep learning of high-dimensional partial differential equations. In Peter Carr
Gedenkschrift: Research Advances in Mathematical Finance, pages 637–655. World Scientific, 2024.

[18] Lorenz Richter, Leon Sallandt, and Nikolas N¨usken. Solving highdimensional parabolic pdes using the tensor train format. In International
Conference on Machine Learning, pages 8998–9009. PMLR, 2021.

[19] Ulrich Schollw¨ock. The density-matrix renormalization group in the age of
matrix product states. Annals of physics, 326(1):96–192, 2011.

[20] Y-Y Shi, L-M Duan, and Guifre Vidal. Classical simulation of quantum many-body systems with a tree tensor network. Physical review a,
74(2):022320, 2006.

[21] G´abor Tak´acs and Domonkos Tikk. Alternating least squares for personalized ranking. In Proceedings of the sixth ACM conference on Recommender
systems, pages 83–90, 2012.

[22] Frank Verstraete and J Ignacio Cirac. Renormalization algorithms for
quantum-many body systems in two and higher dimensions. arXiv preprint
cond-mat/0407066, 2004.

[23] Frank Verstraete and J Ignacio Cirac. Valence-bond states for quantum
computation. Physical Review A, 70(6):060302, 2004.