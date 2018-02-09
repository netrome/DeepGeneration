import settings
import train_stage
import initialize_working_model
import csv
import signal


# Ugly hack to trap Vlad when he appears
# ...Vlad is the name of a bug that appeared during some longer trainings,
#  he causes segmentation faults that crashes the training without any other observable effects
def catch_vlad(signum, frame):
    print(">>> Vlad was here! <<<")
signal.signal(signal.SIGSEGV, catch_vlad)


def update_settings(stage, fade_in, batch_size, learning_rate, chunks, steps):
    print("Updating settings")
    settings.STAGE = stage
    settings.BATCH_SIZE = batch_size
    settings.LEARNING_RATE = learning_rate
    settings.CHUNKS = chunks
    settings.STEPS = steps
    settings.FADE_IN = fade_in
    settings.sync_settings()
    print("Settings updated")

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
if not settings.WORKING_MODEL:
    print("Resetting model")
    initialize_working_model.main()
settings.WORKING_MODEL = True
for conf in config:
    update_settings(*conf)
    print("Calling train_stage main method")
    train_stage.main()


