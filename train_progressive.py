import settings
import train_stage
import initialize_working_model


def update_settings(stage, fade_in, batch_size, learning_rate, chunks, steps):
    settings.STAGE =stage
    settings.BATCH_SIZE = batch_size
    settings.LEARNING_RATE = learning_rate
    settings.CHUNKS = chunks
    settings.STEPS = steps
    settings.FADE_IN = fade_in
    settings.sync_settings()

# Force progressive training
initialize_working_model.main()
for i in range(5):
    update_settings(stage=i+1, fade_in=False, batch_size=16, learning_rate=0.0001, chunks=20, steps=150)
    train_stage.main()
    update_settings(stage=i+1, fade_in=True, batch_size=16, learning_rate=0.0001, chunks=2, steps=150)
    train_stage.main()

update_settings(stage=6, fade_in=False, batch_size=16, learning_rate=0.0001, chunks=200, steps=150)
for i in range(10):
    print("---Perfecting the training---")
    train_stage.main()



