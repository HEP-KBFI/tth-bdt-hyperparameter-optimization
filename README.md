# tth-bdt-hyperparameter-optimization [![Build Status](https://travis-ci.org/HEP-KBFI/tth-bdt-hyperparameter-optimization.svg?branch=master)](https://travis-ci.org/HEP-KBFI/tth-bdt-hyperparameter-optimization)
Evolutionary algorithms for hyperparameter optimization for BDT (XGBoost) and NN (with MNIST numbers dataset for testing).


## Installation

If running with CMSSW:

````console
git clone https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization.git $CMSSW_BASE/src/tthAnalysis/bdtHyperparameterOptimization
cd $CMSSW_BASE/src
scram b -j 8
cd $CMSSW_BASE/src/tthAnalysis/bdtHyperparameterOptimization
pip install -r requirements.txt --user
````
* attrs==19.1.0 version needed due to 19.2.0 breaking with older version of pytest.
[update pytest to pytest==5.2.0](https://stackoverflow.com/questions/58189683/typeerror-attrib-got-an-unexpected-keyword-argument-convert)

Also in order for feature importances to work with NN, eli5 package is needed:

````console
pip install --user eli5
````

### Wiki

For more detailed information visit the [wiki](https://github.com/HEP-KBFI/tth-bdt-hyperparameter-optimization/wiki)


### MNIST numbers dataset

Available here:
[Mnist numbers dataset](http://yann.lecun.com/exdb/mnist/)


### Tests

After installation please run the unittests (in $CMSSW_BASE/src/tthAnalysis/bdtHyperparameterOptimization) with:

````console
pytest test/
````


