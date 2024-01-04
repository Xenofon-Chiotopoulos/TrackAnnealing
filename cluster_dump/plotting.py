
import matplotlib.pyplot as plt
variational =[1,2,3,6,9,12,24,36,48]
time_1 = [2.928,5.26,6.09,12.063,19.651,28.444,81.571, 176.799, 269.197]

non_variational = [1,2,12,24,36, 48]
time_2 = [0.897,0.923,1.179,1.352, 1.77, 1.84]

plt.style.use("ggplot")  
plt.plot(variational,time_1, label='Variational Gates')
plt.plot(non_variational,time_2, label='Non-Variational Gates')
plt.xlabel('Number of Gates')
plt.ylabel('Time(s)')
plt.yscale('log')
plt.show()