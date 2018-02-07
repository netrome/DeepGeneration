test_training_stages_fast:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001
	python train_stage.py --stage 2 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 4 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 6 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001  

overnigt_training_2018-02-06:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 200 --steps 500 --cuda --gp --wip --lr 0.0001 --bs 32
	python train_stage.py --stage 2 --chunks 200 --steps 500 --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 3 --chunks 200 --steps 500 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 4 --chunks 200 --steps 400 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 200 --steps 300 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 200 --steps 200 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 6 --chunks 200 --steps 100 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 101 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 102 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 103 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 104 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 105 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 106 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 107 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 108 --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 109 --cuda --gp --wip --lr 0.0001  
