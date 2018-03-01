import json
import sys
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1], "r"))

plt.hist(data, bins=[i/4 for i in range(100)])

plt.show()
