import json
import sys
import matplotlib.pyplot as plt

file = sys.argv[1]

state = json.load(open(file, "r"))

plt.plot(state["history_real"], label="Real predictions")
plt.plot(state["history_fake"], label="Fake predictions")
plt.axis([100, 200, -10, 75])
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Discriminator output")
plt.show()
