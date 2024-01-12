#! /bin/bash
# training with 7 input lags and 2 output leads. 32 minibatches and 150 epochs

python3 date.py --train --n_input 21 --n_output 2 --batch_size 32 --n_epochs 150  --clase ".B03 Consumption kWh"

