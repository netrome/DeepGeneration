test_training_stages_fast:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001
	python train_stage.py --stage 2 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 4 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 6 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001  
