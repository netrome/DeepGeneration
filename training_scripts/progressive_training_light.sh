python initialize_working_model.py
python train_stage.py --stage 1 --chunks 200 --steps 150 --cuda --gp --wip --lr 0.0001 --bs 16
python train_stage.py --stage 1 --chunks 20  --steps 150 --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
python train_stage.py --stage 2 --chunks 200 --steps 150 --cuda --gp --wip --lr 0.0001 --bs 16
python train_stage.py --stage 2 --chunks 20  --steps 150 --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
python train_stage.py --stage 3 --chunks 200 --steps 100 --cuda --gp --wip --lr 0.0001 --bs 16  
python train_stage.py --stage 3 --chunks 20  --steps 100 --cuda --gp --wip --lr 0.0001 --fade-in
python train_stage.py --stage 4 --chunks 200 --steps 100 --cuda --gp --wip --lr 0.0001 
python train_stage.py --stage 4 --chunks 20  --steps 50  --cuda --gp --wip --lr 0.0001 --fade-in
python train_stage.py --stage 5 --chunks 200 --steps 50  --cuda --gp --wip --lr 0.0001 
python train_stage.py --stage 5 --chunks 20  --steps 50  --cuda --gp --wip --lr 0.0001 --fade-in
python train_stage.py --stage 6 --chunks 200 --steps 50  --cuda --gp --wip --lr 0.0001  
