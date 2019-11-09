#!/bin/bash

LOCATION=$PWD
git pull
cd $CMS_BASE/src
scram b -j 8
cd $LOCATION