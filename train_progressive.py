import settings
import train_stage
import initialize_working_model
import csv


def update_settings(stage, fade_in, batch_size, learning_rate, chunks, steps):
    settings.STAGE =stage
    settings.BATCH_SIZE = batch_size
    settings.LEARNING_RATE = learning_rate
    settings.CHUNKS = chunks
    settings.STEPS = steps
    settings.FADE_IN = fade_in
    settings.sync_settings()

# Get parameters from external file
config = []
with open(settings.CONFIG_PATH) as f:
    reader = csv.reader(f, delimiter=",")
    reader.__next__()  # Ignore header
    for row in reader:
        conf = row
        conf[0] = int(conf[0])
        conf[1] = "true" in conf[1].lower()
        conf[2] = int(conf[2])
        conf[3] = float(conf[3])
        conf[4] = int(conf[4])
        conf[5] = int(conf[5])
        config.append(row)

# Force progressive training
initialize_working_model.main()
settings.WORKING_MODEL = True
for conf in config:
    update_settings(*conf)
    train_stage.main()
"""
for i in range(5):
    update_settings(stage=i+1, fade_in=False, batch_size=16, learning_rate=0.0001, chunks=200, steps=150)
    train_stage.main()
    update_settings(stage=i+1, fade_in=True, batch_size=16, learning_rate=0.0001, chunks=20, steps=150)
    train_stage.main()
"""

"""
update_settings(stage=6, fade_in=False, batch_size=16, learning_rate=0.0001, chunks=200, steps=150)
for i in range(10):
    print("---Perfecting the training---")
    train_stage.main()
"""


