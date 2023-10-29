"""
EXERCISE 02: RANKING BINARY NUMBERS WITH A PERCEPTRON
AUTHOR: DAVID VILLA BLANCO
DATE: 09/10/23
SUBJECT: STATISTICAL PHYSICS AND MACHINE LEARNING
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# VARIABLES AND ARRAYS
N = 20 # Length of the input array
I = 1e4 # Number of repetitions of the experiment
J = np.zeros(N)

# FUNCTIONS
def dec_to_bin(n):
	# n: array containing digit by digit the binary number
	N = np.size(n)
	numb = 0
	for i in range(N):
		numb+=n[i]*2**(N-1-i)
	return numb

def perceptron(S, J):
	return np.sign(np.sum(S*J))

def training(P, P_num, P_perfect, J, N=20):
	fails = 1
	while fails > 0:
		fails = 0
		for i in range(P_num):
			n_T = P[i][0]
			n_B = P[i][1]
			n_T_bin = "{0:0{1}b}".format(n_T, int(N/2))
			n_B_bin = "{0:0{1}b}".format(n_B, int(N/2))
			S = np.array([int(n_T_bin[i]) for i in range(int(N/2))])
			S = np.append(S, np.array([int(n_B_bin[i]) for i in range(int(N/2))]))
			S = np.where(S == 0, -1, S)
			value = perceptron(S, J)
			if value == P_perfect[i]:
				continue
			else:
				for j in range(N):
					J[j] += (1/np.sqrt(N))*P_perfect[i]*S[j]
				fails += 1
		eps = fails/P_num
	return J

# PERFECT PERCEPTRON ASSIGNING TO THE SYNAPTIC WEIGTHS
Jperfect = np.zeros(N)
for i in range(1,N+1):
	if 1<=i<=10:
		Jperfect[i-1] = 2**(10-i)
	else:
		Jperfect[i-1] = -Jperfect[i-11]

# WE GENERATE THE PERFECT INPUT
fails = 0
for i in range(int(I)):
	n_T = np.random.randint(0, 2**10)
	n_B = np.random.randint(0, 2**10)
	n_T_bin = "{0:0{1}b}".format(n_T, int(N/2))
	n_B_bin = "{0:0{1}b}".format(n_B, int(N/2))
	S = np.array([int(n_T_bin[i]) for i in range(int(N/2))])
	S = np.append(S, np.array([int(n_B_bin[i]) for i in range(int(N/2))]))
	S = np.where(S == 0, -1, S)
	value_perfect = perceptron(S,Jperfect)

	if value_perfect == 1 and n_T>n_B:
		continue
	elif value_perfect == -1 and n_T<=n_B:
		continue
	else:
		fails+=1
print("The perfect perceptron had a error ratio (compared to real value) of: " + str(100*fails/I) + "%")

# WE GENERATE THE GAUSSIAN INPUT
fails = 0
for i in range(int(I)):
	n_T = np.random.randint(0, 2**10)
	n_B = np.random.randint(0, 2**10)
	n_T_bin = "{0:0{1}b}".format(n_T, int(N/2))
	n_B_bin = "{0:0{1}b}".format(n_B, int(N/2))
	S = np.array([int(n_T_bin[i]) for i in range(int(N/2))])
	S = np.append(S, np.array([int(n_B_bin[i]) for i in range(int(N/2))]))
	S = np.where(S == 0, -1, S)
	value_perfect = perceptron(S, Jperfect)
	value_norm = perceptron(S, np.random.normal(0, 1, N))
	if value_norm == value_perfect:
		continue
	else:
		fails+=1
print("The gaussian perceptron had a error ratio (compared to the perfect perceptron) of: " + str(100*fails/I) + "%")

# PERCEPTRON TRAINING
# Data sets
P1_num = 500
P2_num = 2000
P1 = np.random.randint(0, 2**10, size=(P1_num,2))
P1_perfect = np.zeros(P1_num)
P2 = np.random.randint(0, 2**10, size=(P2_num,2))
P2_perfect = np.zeros(P2_num)
for i in range(np.shape(P1)[0]):
	if P1[i][0]>P1[i][1]:
		P1_perfect[i] = 1
	else:
		P1_perfect[i] = -1
for i in range(np.shape(P2)[0]):
	if P2[i][0]>P2[i][1]:
		P2_perfect[i] = 1
	else:
		P2_perfect[i] = -1
# Training
J01 = np.random.normal(0, 1, N)
J01_initial = np.copy(J01)
J02 = np.random.normal(0, 1, N)
J02_initial = np.copy(J02)
# J01 Training
J01 = training(P1, P1_num, P1_perfect, J01)
# J02 Training
J02 = training(P2, P2_num, P2_perfect, J02)

# Plot for evolution
ax1 = plt.subplot(2,1,1)
plt.plot(np.arange(np.size(J01)), np.sign(Jperfect*J01)*np.log2(np.abs(J01)), label = r"J$_{P=500}$ evolution")
plt.ylabel(r"sign(J$^{*}$J)log$_2$(|J|)")
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
ax2 = plt.subplot(2,1,2)
plt.plot(np.arange(np.size(J02)), np.sign(Jperfect*J02)*np.log2(np.abs(J02)), label = r"J$_{P=2000}$ evolution")
plt.xlabel("index of the synaptic weight")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel(r"sign(J$^{*}$J)log$_2$(|J|)")
plt.legend()
plt.savefig("/Users/davidvillablanco/Desktop/UNIROMA/Primo semestre/Altre/SPandML/Exercises/02/Ex02_Villa(Evolution_Plots).jpg", dpi=300)
plt.show()

# Error of new trained perceptrons
# P = 500
fails = 0
for i in range(int(I)):
	n_T = np.random.randint(0, 2**10)
	n_B = np.random.randint(0, 2**10)
	n_T_bin = "{0:0{1}b}".format(n_T, int(N/2))
	n_B_bin = "{0:0{1}b}".format(n_B, int(N/2))
	S = np.array([int(n_T_bin[i]) for i in range(int(N/2))])
	S = np.append(S, np.array([int(n_B_bin[i]) for i in range(int(N/2))]))
	S = np.where(S == 0, -1, S)
	value_perfect = perceptron(S, Jperfect)
	value_trained1 = perceptron(S, J01)
	if value_trained1 == value_perfect:
		continue
	else:
		fails+=1
print("The gaussian trained perceptron (P = 500) had a error ratio (compared to the perfect perceptron) of: " + str(100*fails/I) + "%")
# P = 2000
fails = 0
for i in range(int(I)):
	n_T = np.random.randint(0, 2**10)
	n_B = np.random.randint(0, 2**10)
	n_T_bin = "{0:0{1}b}".format(n_T, int(N/2))
	n_B_bin = "{0:0{1}b}".format(n_B, int(N/2))
	S = np.array([int(n_T_bin[i]) for i in range(int(N/2))])
	S = np.append(S, np.array([int(n_B_bin[i]) for i in range(int(N/2))]))
	S = np.where(S == 0, -1, S)
	value_perfect = perceptron(S, Jperfect)
	value_trained2 = perceptron(S, J02)
	if value_trained2 == value_perfect:
		continue
	else:
		fails+=1
print("The gaussian trained perceptron (P = 2000) had a error ratio (compared to the perfect perceptron) of: " + str(100*fails/I) + "%")
