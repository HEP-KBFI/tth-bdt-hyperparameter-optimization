#!/bin/bash

LOCATION=$PWD
git pull
cd ~/CMSSW_10_2_10/src
scram b -j 8
cd $LOCATION