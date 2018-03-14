test_training_stages_fast:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001
	python train_stage.py --stage 2 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 4 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001   
	python train_stage.py --stage 6 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001  

test_fade_in_complete:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 1 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
	python train_stage.py --stage 2 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 2 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 3 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 4 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 4 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 5 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 6 --chunks 2 --steps 5 --cuda --gp --wip --lr 0.0001  

progressive_training_light:
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

progressive_training_medium:
	python initialize_working_model.py
	python train_stage.py --stage 1 --chunks 200 --steps 500  --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 1 --chunks 20  --steps 500  --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
	python train_stage.py --stage 2 --chunks 200 --steps 500  --cuda --gp --wip --lr 0.0001 --bs 16
	python train_stage.py --stage 2 --chunks 20  --steps 500  --cuda --gp --wip --lr 0.0001 --bs 16 --fade-in
	python train_stage.py --stage 3 --chunks 200 --steps 400  --cuda --gp --wip --lr 0.0001 --bs 16  
	python train_stage.py --stage 3 --chunks 20  --steps 400  --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 4 --chunks 200 --steps 400  --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 4 --chunks 20  --steps 200  --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 5 --chunks 200 --steps 200  --cuda --gp --wip --lr 0.0001 
	python train_stage.py --stage 5 --chunks 20  --steps 100  --cuda --gp --wip --lr 0.0001 --fade-in
	python train_stage.py --stage 6 --chunks 200 --steps 101  --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 102  --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 103  --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 104  --cuda --gp --wip --lr 0.0001  
	python train_stage.py --stage 6 --chunks 200 --steps 105  --cuda --gp --wip --lr 0.0001  

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

overnight_vae_2018-02-19:
	python initialize_working_model.py
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 100 --wip

overnight_vae_long:
	python initialize_working_model.py
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip

overnight_vae_real:
	python initialize_working_model.py
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data
	python -u vae_training.py --cuda --chunks 200 --steps 200 --wip --real-data

overnight_aegan:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip  # Karras run resumed from here
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip

aegan_cont:
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip

aegan_debug:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip 
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip 
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip 
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip 

aegan_helen_debug:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 100 --steps 20 --wip --helen-data

overnight_aegan_real:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip --real-data

overnight_aegan_helen:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip --helen-data
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip --helen-data

aegan_real_cont:
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 200 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 201 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 202 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 203 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 204 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 205 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 206 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 207 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 208 --steps 100 --wip --real-data
	python -u aegan_training.py --cuda --chunks 209 --steps 100 --wip --real-data

aegan_long:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip

aegan_real_long:
	python initialize_working_model.py
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 400 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 401 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 402 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 403 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 404 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip --real-data
	python -u aegan_training.py --cuda --chunks 405 --steps 200 --wip --real-data

regressor:
	python initialize_working_model.py
	python -u regressor_training.py --cuda --chunks 50 --steps 100 --wip $(extra) 
	python -u regressor_training.py --cuda --chunks 50 --steps 100 --wip $(extra) 
	python -u regressor_training.py --cuda --chunks 50 --steps 100 --wip $(extra) 
	python -u regressor_training.py --cuda --chunks 50 --steps 100 --wip $(extra) 
