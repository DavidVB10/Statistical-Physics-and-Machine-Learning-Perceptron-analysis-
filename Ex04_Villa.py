import numpy as np
import matplotlib.pyplot as plt

# Errors obtained for 1000 training and testing for N1 = 20 and N2 = 40
err20_mean = np.array([0.491432, 0.416564, 0.348245, 0.21106, 0.119281, 0.081129, 0.060547])
err20_bar = np.array([0.07650694, 0.0689052, 0.06343432, 0.04685965, 0.0288981, 0.0228744, 0.01673929])
err40_mean = np.array([0.493796, 0.488817, 0.415483, 0.344571, 0.212951, 0.120184, 0.082426])
err40_bar = np.array([0.05618497, 0.0529061, 0.0511469, 0.04700429, 0.0348535, 0.02210118, 0.01687686])
# Various variables
N1 = 20
N2 = 40
P1 = np.array([1,10,20,50,100,150,200])
P2 = np.array([1,2,20,40,100,200,300])
t = np.linspace(0,0.5,100)
I = np.array([int(5*1e2),int(5*1e2),50])
I_dib1 = np.arange(I[0])
I_dib2 = np.arange(I[1])
I_dib3 = np.arange(I[2])
alph = 5
eps_0 = 0.25
a = np.array([0.5,0.9,0.1])
alpha_array = np.linspace(0,10,1000)

def alpha(eps):
	return (1-eps)*np.pi*(1/np.tan(np.pi*eps))

def f1(eps, alph):
	return 1-alph*np.tan(np.pi*eps)/np.pi

def f2(eps, alph):
	return np.arctan(np.pi*(1-eps)/alph)/np.pi

def eps_i(alph, eps0, a, N, f_num):
	array = np.array([eps0])
	for i in range(N-1):
		if f_num == 1:
			array = np.append(array, (1-a)*array[i]+a*f1(array[i],alph))
		elif f_num == 2:
			array = np.append(array, (1-a)*array[i]+a*f2(array[i],alph))
	return array

def eps_iterative(alpha, eps0, a, f_num):
	array = np.array([eps0])
	for i in range(len(alpha)-1):
		if f_num == 1:
			array = np.append(array, (1-a)*array[i]+a*f1(array[i],alpha[i]))
		elif f_num == 2:
			array = np.append(array, (1-a)*array[i]+a*f2(array[i],alpha[i]))
	return array

plt.figure()
plt.plot(P1/N1, err20_mean, "ro", label = "N = " + str(N1))
plt.plot(P2/N2, err40_mean, "bo", label = "N = " + str(N2))
plt.errorbar(P1/N1, err20_mean, err20_bar, linestyle="None")
plt.errorbar(P2/N2, err40_mean, err40_bar, linestyle="None")
plt.plot(alpha(t), t, color="black", label="Theoretical curve (parametric)")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\epsilon$")
plt.xlim([-0.5,10.5])
plt.legend()
plt.show()
# PLOTS FOR F1
plt.subplot(2,2,1)
plt.plot(I_dib1, eps_i(alph, eps_0, a[0], I[0], 1), color="black", label=(r"$\alpha$ "+ "= " + str(a[0])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.subplot(2,2,2)
plt.plot(I_dib2, eps_i(alph, eps_0, a[1], I[1], 1), color="black", label=(r"$\alpha$ "+ "= " + str(a[1])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.subplot(2,2,(3,4))
plt.plot(I_dib3, eps_i(alph, eps_0, a[2], I[2], 1), color="black", label=(r"$\alpha$ "+ "= " + str(a[2])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.show()
# PLOTS FOR F2
plt.subplot(2,2,1)
plt.plot(I_dib3, eps_i(alph, eps_0, a[0], I[2], 2), color="black", label=(r"$\alpha$ "+ "= " + str(a[0])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.subplot(2,2,2)
plt.plot(I_dib3, eps_i(alph, eps_0, a[1], I[2], 2), color="black", label=(r"$\alpha$ "+ "= " + str(a[1])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.subplot(2,2,(3,4))
plt.plot(I_dib3, eps_i(alph, eps_0, a[2], I[2], 2), color="black", label=(r"$\alpha$ "+ "= " + str(a[2])))
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.legend()
plt.show()
# PLOTS FOR F1 AND F1 DIFFERENTS eps_0, a, alpha
plt.figure()
plt.plot(alpha_array, eps_iterative(alpha_array, 0.5, a[2], 2), color="black")
plt.xlabel("i")
plt.ylabel(r"$\epsilon_i$")
plt.show()
# POINTS AND FITS
plt.figure()
plt.plot(P1/N1, err20_mean, "ro", label = "N = " + str(N1))
plt.plot(P2/N2, err40_mean, "bo", label = "N = " + str(N2))
plt.errorbar(P1/N1, err20_mean, err20_bar, linestyle="None")
plt.errorbar(P2/N2, err40_mean, err40_bar, linestyle="None")
plt.plot(alpha(t), t, color="black", label="Theoretical curve (parametric)")
plt.plot(alpha_array, eps_iterative(alpha_array, 0.5, 0.9, 2), color="red", label="Theoretical curve (iterative)")
plt.xlabel("P/N")
plt.ylabel(r"$\epsilon$")
plt.xlim([-0.5,10.5])
plt.legend()
plt.show()