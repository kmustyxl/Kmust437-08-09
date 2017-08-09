from AssignRule import *
import matplotlib.pyplot as plt
import time
start_time = time.clock()
the_best, the_worst = Bayes_net(pop_gen, ls_frequency)
end_time = time.clock()
run_time = end_time - start_time
gen = [i for i in range(pop_gen)]
fig = plt.figure()
for i, factory in enumerate(range(num_factory)):
    ax = fig.add_subplot(1,num_factory,i+1)
    ax.plot(gen, the_best[i], 'b',label = '$Best$')
    ax.plot(gen, the_worst[i], 'r',label = '$Worst$')
    ax.set_xlabel(r'gen')
    ax.set_ylabel(r'fitness')
    ax.set_title(r'Factory:%s'%i)
    ax.grid()
    ax.legend()
fig.suptitle('$Distributed$'+' '+'$flowshop$'+' '+'$scheduling$'+' '+'$problem$'+'\nRun:%.2fs'%run_time)
plt.show()