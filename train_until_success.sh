#!/bin/bash
python initialize_working_model.py
until python train_progressive.py --wip --cuda --gp --config $1
do
  echo "--> Restarting script <--"
done

