import numpy as np
import matplotlib.pyplot as plt

N1 = 20
N2 = 40
P_test_num = 1000
P1 = np.array([1,10,20,50,100,150,200])
P2 = np.array([1,2,20,40,100,200,300])

def teacher_syn_coup(N):
    i = np.arange(N)
    condition = (i >= 0) & (i <= (N // 2 - 1))
    T = np.zeros(N)
    T[condition] = 2 ** (N // 2 - i[condition] - 1)
    T[~condition] = -T[i[~condition] - N // 2]
    return T

def sign(V):
	return np.where(V > 0, 1,-1)

T20 = teacher_syn_coup(20)
T40 = teacher_syn_coup(40)

S_vector20 = np.array([])
err20 = np.array([])
S_vector40 = np.array([])
err40 = np.array([])

for i in range(P_test_num):
	print(i)
	for N in [N1,N2]:
		if N == 20:
			for P_num in P1:
				S = np.random.normal(0,1,N)
				P = np.random.choice([-1,1], (P_num, N))
				sigma_T = sign(np.sum(P*T20, axis=1))
				sigma_S = sign(np.sum(P*S, axis=1))
				mismatch = sigma_T!=sigma_S
				while np.sum(mismatch)!=0:
					S += np.sum((1/np.sqrt(N))*(mismatch*sigma_T)[:, np.newaxis]*P*(1+np.random.normal(0,np.sqrt(50), np.shape(P))), axis=0)
					sigma_S = sign(np.sum(P*S, axis=1))
					mismatch = sigma_T!=sigma_S
				S_vector20 = np.append(S_vector20, S)
			S_vector20 = S_vector20.reshape(7, N)
			P_test = np.random.choice([-1,1], (P_test_num, N))
			sigma_T = sign(np.sum(P_test*T20, axis=1))
			sigma_Svector20 = sign(np.sum(P_test*S_vector20[:, np.newaxis], axis=2))
			mismatch = sigma_T!=sigma_Svector20
			mismatch_sum = (1/P_test_num)*np.sum(mismatch, axis=1)
			err20 = np.append(err20, mismatch_sum)
		else:		
			for P_num in P2:
				S = np.random.normal(0,1,N)
				P = np.random.choice([-1,1], (P_num, N))
				sigma_T = sign(np.sum(P*T40, axis=1))
				sigma_S = sign(np.sum(P*S, axis=1))
				mismatch = sigma_T!=sigma_S		
				while np.sum(mismatch)!=0:
					S += np.sum((1/np.sqrt(N))*(mismatch*sigma_T)[:, np.newaxis]*P*(1+np.random.normal(0,np.sqrt(50), np.shape(P))), axis=0)
					sigma_S = sign(np.sum(P*S, axis=1))
					mismatch = sigma_T!=sigma_S
				S_vector40 = np.append(S_vector40, S)
			S_vector40 = S_vector40.reshape(7, N)
			P_test = np.random.choice([-1,1], (P_test_num, N))
			sigma_T = sign(np.sum(P_test*T40, axis=1))
			sigma_Svector40 = sign(np.sum(P_test*S_vector40[:, np.newaxis], axis=2))
			mismatch = sigma_T!=sigma_Svector40
			mismatch_sum = (1/P_test_num)*np.sum(mismatch, axis=1)
			err40 = np.append(err40, mismatch_sum)
	S_vector20 = np.array([])
	S_vector40 = np.array([])
err20 = err20.reshape(P_test_num, 7)
err40 = err40.reshape(P_test_num, 7)

err20_mean = np.mean(err20, axis=0)
err20_bar = np.std(err20, axis = 0)
err40_mean = np.mean(err40, axis=0)
err40_bar = np.std(err40, axis = 0)

plt.figure()
plt.plot(P1, err20_mean, "ro", label = "N = " + str(N1))
plt.plot(P2, err40_mean, "bo", label = "N = " + str(N2))
plt.errorbar(P1, err20_mean, err20_bar, linestyle="None")
plt.errorbar(P2, err40_mean, err40_bar, linestyle="None")
plt.xlabel("P")
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.show()

plt.figure()
plt.plot(P1/N1, err20_mean, "ro", label = "N = " + str(N1))
plt.plot(P2/N2, err40_mean, "bo", label = "N = " + str(N2))
plt.errorbar(P1/N1, err20_mean, err20_bar, linestyle="None")
plt.errorbar(P2/N2, err40_mean, err40_bar, linestyle="None")
plt.xlabel("P/N")
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.show()