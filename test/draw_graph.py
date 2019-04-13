import pickle
import matplotlib.pyplot as plt 

file = open("scores.pkl", "rb")
a = pickle.load(file)
file.close()

b = []
c = []

for x, y in a:
	b.append(x)
	c.append(y)

plt.plot(b, c, 'r--')
plt.show()