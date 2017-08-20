'''
Green_scheduling

2017.08.07
Author: Yang Xiaolin
'''
from AssignRule import *
from random import choice
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
V = [1, 1.3, 1.55, 1.75, 2.10]
v = np.zeros((num_job, num_machine))
for i in range(num_machine):
    for j in range(num_job):
        temp = choice(V)
        v[j][i] = temp
def Green_Calcfitness(n, m, sort, test_data, v):
    c_time1 = np.zeros([n, m])
    c_time1[0][0] = test_data[sort[0]][0] / v[0][0]
    for i in range(1, n):
        c_time1[i][0] = c_time1[i - 1][0] + test_data[sort[i]][0] / v[i][0]
    for i in range(1, m):
        c_time1[0][i] = c_time1[0][i - 1] + test_data[0][i] / v[0][i]
    for i in range(1, n):
        for k in range(1, m):
            c_time1[i][k] = test_data[sort[i]][k] / v[i][k] + max(c_time1[i - 1][k], c_time1[i][k - 1])
    return c_time1[n - 1][m - 1]

def TCE(n, m, sort, test_data,v):
    per_consumption_Standy = 1
    standy_time = 0
    Energy_consumption = 0
    for k in range(m):
        for i in range(n):
            per_consumption_V = 4 * v[i][k] * v[i][k] # 机器加速单位时间能源消耗
            Energy_consumption += test_data[sort[i]][k] / v[i][k] * per_consumption_V
    C_time = Green_Calcfitness(n,m,sort, test_data, v)
    for k in range(m):
        Key_time = C_time
        for i in range(n):
            Key_time -= test_data[sort[i]][k] / v[i][k]
        standy_time += Key_time
    Energy_consumption += standy_time * per_consumption_Standy
    return C_time, Energy_consumption


def green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data):
    #每个工厂的工件数
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    Mat_pop =[[[0 for i in range(len_job[k] + 2) ]for j in range(1000)] for k in range(num_factory)] #最后两个元素分别是经济指标和绿色指标
    non_dominated_pop = [[] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(1000):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k], Mat_pop[i][j][k+1]= TCE(len_job[i], num_machine, sort, test_data,v)
                else:
                    Mat_pop[i][j][k] = sort[k]
        Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2])
        Mat_pop[i] = Mat_pop[i][0:100]
        for j in range(100):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(100):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                non_dominated_pop[i].append(Mat_pop[i][j])
    return Mat_pop[:][0:100], non_dominated_pop

def select_non_dominated_pop(num_factory, factory_job_set, Mat_pop):
    temp_non_dominated_pop = [[] for k in range(num_factory)]
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    for i in range(num_factory):
        for j in range(len(Mat_pop[i])):
            compare_fitness1 = Mat_pop[i][j][len_job[i]]
            compare_fitness2 = Mat_pop[i][j][len_job[i]+1]
            b_non_dominated = True
            for k in range(len(Mat_pop[i])):
                if j != k:
                    if Mat_pop[i][k][len_job[i]] <= compare_fitness1 and Mat_pop[i][k][len_job[i]+1] <= compare_fitness2:
                        if Mat_pop[i][k][len_job[i]] < compare_fitness1 or Mat_pop[i][k][len_job[i]+1] < compare_fitness2:
                            b_non_dominated = False
                            break
            if b_non_dominated == True:
                temp_non_dominated_pop[i].append(Mat_pop[i][j])
    return temp_non_dominated_pop


def Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop):
    # 每个工厂每次更新20个个体
    newpop = [[[-1 for i in range(len(factory_job_set[k]) + 2)] for j in range(100)] for k in range(num_factory)]
    # 三维数组--第一维：工厂数；第二维：每个工厂的所有相邻关系数组；第三维：与上一个工件的关系
    prob_mat = [[[0.0 for i in range(len(factory_job_set[k]))]
                 for l in range(len(factory_job_set[k]) - 1)] for k in range(num_factory)]
    for i in range(num_factory):
        for num in range(100):
            temp = 0.0
            job_len = len(factory_job_set[i])
            # 用轮盘赌确定第一个工件
            # temp_prob_mat = prob_mat[[[[:]:]:]:]
            r = random.random()
            if r < prob_mat_first[i][0]:
                newpop[i][num][0] = factory_job_set[i][0]
            else:
                j = 0
                temp += prob_mat_first[i][j]
                while temp < r:
                    j += 1
                    temp += prob_mat_first[i][j]
                newpop[i][num][0] = factory_job_set[i][j]
            for k in range(1, job_len):
                temp_job = []
                for ii in Mat_pop[i][0:100]:
                    if ii[k] in newpop[i][num]:
                        continue
                    elif ii[k - 1] == newpop[i][num][k - 1]:
                        temp_job.append(ii[k])
                if len(temp_job) == 0:
                    key = True
                    while key:
                        newpop_temp = choice(factory_job_set[i])
                        if newpop_temp not in newpop[i][num]:
                            newpop[i][num][k] = newpop_temp
                            key = False
                    continue
                for m in range(job_len):
                    prob_mat[i][k - 1][m] = temp_job.count(factory_job_set[i][m]) / len(temp_job)
                r = random.random()
                temp = 0.0
                j = 0
                while temp < r:
                    temp += prob_mat[i][k - 1][j]
                    j += 1
                    if j == len(factory_job_set[i]):
                        break
                newpop[i][num][k] = factory_job_set[i][j - 1]
            newpop[i][num][-2], newpop[i][num][-1] = TCE(len(factory_job_set[i]), num_machine, newpop[i][num], test_data, v)
        newpop[i] = sorted(newpop[i], key= lambda x:x[-2])
    return newpop[:][0:20]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def green_local_search(newpop, ls_frequency, factory_job_set, num_factory):
    ls_pop = [[[[-1 for i in range(len(factory_job_set[k]))] for j in range(ls_frequency)] for l in range(20)]for k in range(num_factory)]
    select_ls_pop = [[[-1 for i in range(len(factory_job_set[k]))]  for l in range(20)]for k in range(num_factory)]
    for i in range(num_factory):
        for l in range(20):
            ls_pop[i][l][0][:] = newpop[i][l][:]
            for j in range(1, ls_frequency):
                temp1 = np.random.randint(0, len(factory_job_set[i]))
                temp2 = np.random.randint(0, len(factory_job_set[i]))
                while temp1 == temp2:
                    temp1 = np.random.randint(0, len(factory_job_set[i]))
                    temp2 = np.random.randint(0, len(factory_job_set[i]))
                temp = newpop[i][l][temp1]
                newpop[i][l][temp1] = newpop[i][l][temp2]
                newpop[i][l][temp2] = temp
                ls_pop[i][l][j][:-2] = newpop[i][l][:-2]
                newpop[i][l][:-2] = ls_pop[i][l][0][:-2]
                ls_pop[i][l][j][-2],ls_pop[i][l][j][-1] = TCE(len(factory_job_set[i]), num_machine, ls_pop[i][l][j][0:-2], test_data, v)
            select_ls_pop[i][l] = sorted(ls_pop[i][l],key= lambda x:x[-2])[0]
        select_ls_pop[i] = sorted(select_ls_pop[i],key= lambda x:x[-2])
    return select_ls_pop

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Green_Bayes_net(pop_gen, ls_frequency):
    fitness = [[0 for i in range(20)] for j in range(num_factory)]
    green_fitness = [[0 for i in range(20)] for j in range(num_factory)]
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory)
    Mat_pop,  non_dominated_pop= green_initial_Bayes(num_machine, num_factory, factory_job_set, test_data)
    min_fitness = [[-1 for j in range(20)] for i in range(num_factory)]
    min_fitness_green = [[-1 for j in range(20)] for i in range(num_factory)] #最小fitness对应的能源消耗
    min_index = [[-1 for j in range(20)]  for i in range(num_factory)]
    the_best = [[[0,0] for i in range(pop_gen)] for j in range(num_factory)]
    the_worst = [[[0,0] for i in range(pop_gen)] for j in range(num_factory)]
    temp_list = []
    Pareto_num = [[0 for i in range(pop_gen)] for j in range(num_factory)]
    data = [[] for i in range(num_factory)]
    Pareto_gen = [[[] for i in range(pop_gen) ]for k in range(num_factory)]
    for gen in range(pop_gen):
        prob_mat_first = Bayes_update(Mat_pop, factory_job_set, num_factory)
        newpop = Green_New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop)
        ls_pop = green_local_search(newpop, ls_frequency, factory_job_set, num_factory)
        # for i in range(num_factory):
        #     for j in range(20):
        #         fitness[i][j] = ls_pop[i][]
        #         green_fitness[i][j] = TCE(len(factory_job_set[i]), num_machine, ls_pop[i][j], test_data, v)
        #     for j in range(20):
        #         min_index[i][j] = np.argsort(fitness[i][:])[0]
        #         min_fitness[i][j] = fitness[i][min_index[i][j]]
        #         min_fitness_green[i][j] = green_fitness[i][min_index[i][j]]
        for i in range(num_factory):
            temp = len(factory_job_set[i])
            k_index = -1
            for k in range(19,-1,-1):
                k_index += 1
                for j in range(99-k_index,-1,-1):
                    if float(ls_pop[i][k][-2]) < float(Mat_pop[i][j][temp]) or float(ls_pop[i][k][-1]) < float(Mat_pop[i][j][temp+1]):
                        temp_list = ls_pop[i][k]
                        for l in range(len(factory_job_set[i]) + 2):
                            Mat_pop[i][j][l] = temp_list[l]
                        temp_list = []
                        break
            Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-2])

        temp_non_dominated = select_non_dominated_pop(num_factory, factory_job_set, Mat_pop)
        for i in range(num_factory):
            for j in range(len(temp_non_dominated[i])):
                non_dominated_pop[i].append(temp_non_dominated[i][j])
        non_dominated_pop = select_non_dominated_pop(num_factory, factory_job_set, non_dominated_pop)
        for i in range(num_factory):
            for individual in non_dominated_pop[i]:
                if individual not in Pareto_gen[i][gen]:
                    Pareto_gen[i][gen].append(individual)
            Pareto_num[i][gen] = len(Pareto_gen[i][gen])
        non_dominated_pop = [[] for k in range(num_factory)]
            #print('代数：%s'%gen,Pareto_gen[0])
            # the_best[i][gen][0] = Mat_pop[i][0][temp]
            # the_worst[i][gen][0] = Mat_pop[i][-1][temp]
            # the_best[i][gen][1] = Mat_pop[i][0][temp+1]
            # the_worst[i][gen][1] = Mat_pop[i][-1][temp+1]
    return the_best,the_worst, Pareto_num, Pareto_gen
start_time = time.clock()
the_best, the_worst ,Pareto_num, Pareto_gen= Green_Bayes_net(pop_gen, ls_frequency)
end_time = time.clock()
run_time = end_time - start_time
gen = [i for i in range(pop_gen)]
X = [[[]for j in range(2)]for i in range(num_factory)]
Y = [[[]for j in range(2)]for i in range(num_factory)]
for i in range(num_factory):
    for j in range(2):
        for l in range(len(Pareto_gen[i][j*(pop_gen-1)])):
            X[i][j].append(int(Pareto_gen[i][j*(pop_gen-1)][l][-2]))
            Y[i][j].append(int(Pareto_gen[i][j*(pop_gen-1)][l][-1]))
print(X[0][0])
print(Y[0][0])
fig = plt.figure()
#for i, factory in enumerate(range(num_factory)):
ax = fig.add_subplot(111)
# for j in range(Pareto_num[i][0]):
ax.scatter(X[0][0], Y[0][0])
#ax.scatter(Pareto_gen[i][-1][:][-2], Pareto_gen[i][-1][:][-1],'gD')
ax.set_xlabel(r'gen')
ax.set_ylabel(r'fitness')
ax.set_title(r'Factory')
ax.grid()
ax.legend()
#fig.suptitle('$Distributed$'+' '+'$flowshop$'+' '+'$scheduling$'+' '+'$problem$'+'\nRun:%.2fs'%run_time)
plt.show()

# temp1 = []
# temp2 = []
# for i in range(pop_gen):
#     temp1.append(the_best[0][i][0])
#     temp2.append(the_best[0][i][1])
# ax = Axes3D(fig)
# for i in gen:
#    # for j in Pareto_num[0]:
#         for k in range(Pareto_num[0][i]):
#             ax.scatter(i, Pareto_gen[0][i][k][-2], Pareto_gen[0][i][k][-1],'r')
# ax.set_xlabel(r'gen')
# ax.set_ylabel(r'fitness')
# ax.set_zlabel(r'Carbon')
# ax.legend()
# ax.set_title(r'Multive—object')
# plt.show()
# print(Pareto_num)
#print(Pareto_gen[0][-1])
#print(Pareto_gen[1][-1])
