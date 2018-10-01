# Section 2.3
### tune the hyper param of epochs
import matplotlib.pyplot as plt
epochs = [5,10,20,30]
real = 4812
real_std = 75
epochs = [4, 8, 12, 16, 20]
means = [3407, 3769, 4453, 3748, 4303]
stds = [1020, 909, 538, 183, 102]
plt.clf()
plt.fill_between([0,24],[real-real_std, real-real_std], [real+real_std, real+real_std], color='r', alpha=0.2)
plt.plot([0,24],[real,real],ls='-',lw=2,color='r',label='expert')
plt.errorbar(epochs, means, yerr=stds, fmt='o', capsize=2,label='behavior cloning')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('position')
plt.savefig('bc.png',format='png')
plt.show()

# Section 3.2:
real = 3778
real_std = 4
bc = 993
bc_std = 303
means= [993, 945, 3239, 2700, 2423, 3777, 3769, 3776, 3786, 3781]
stds=[303, 421, 590, 824, 928, 8, 18, 3, 4.2, 3.5]

plt.clf()
plt.fill_between([0,24],[real-real_std, real-real_std], [real+real_std, real+real_std], color='r', alpha=0.2)
plt.plot([0,24],[real,real],ls='-',lw=2,color='r',label='expert')
plt.errorbar(1, bc, yerr=bc_std, fmt='o', capsize=2, label='behavior cloning')
plt.errorbar(np.arange(9)+2, means[1:], yerr=stds[1:], fmt='o', capsize=2,label='DAgger')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('position')
plt.title("Hopper imitation")
plt.savefig('dagger.png',format='png')
plt.show()
