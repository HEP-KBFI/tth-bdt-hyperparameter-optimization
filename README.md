# tth-bdt-hyperparameter-optimization [![Build Status](https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization.svg?branch=master)](https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization)
Evolutionary algorithms for hyperparameter optimization for BDT (XGBoost) and NN (with MNIST numbers dataset for testing).


## Installation

If running with CMSSW:

````console
git clone https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization.git $CMSSW_BASE/src/tthAnalysis/bdtHyperparameterOptimization
pip install pathlib --user
pip install timeout-decorator --user
pip install docopt --user
````


## Particle Swarm Optimization (PSO)

<img src="README/pso_overview1.png" alt="Sensitivity" width="400"/>
<img src="README/pso_overview2.png" alt="Sensitivity" width="400"/>


Calculating the speed for next iteration:
<img src="http://bit.ly/2VvaTvW" align="center" border="0" alt="V_{t+1} = w*V_t + c_1*r_1*(P_t - X_t) + c_2*r_2*(G_t - X_t)" width="449" height="18" />

c1 = c2 for balancing exploration and exploitation


Calculating the position for the next iteration:
<img src="http://bit.ly/2BatXXn" align="center" border="0" alt="X_{t+1} = X_t + V_{t}" width="118" height="18" />


### MNIST numbers dataset

Available here:
[Mnist numbers dataset](http://yann.lecun.com/exdb/mnist/)


