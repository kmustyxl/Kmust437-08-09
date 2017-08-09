# -*- coding:utf-8 -*-
import LoadData as ld
import numpy as np
import random
from random import choice


num_job = int(input('请输入工件总数： '))
num_machine = int(input('请输入单工厂机器总数： '))
num_factory = int(input('请输入工厂总数： '))
ls_frequency = int(input('请输入局部搜索次数： '))
pop_gen = int(input('请输入进化代数： '))
test_data = ld.LoadData(num_job, num_machine)
# def Mideng(li):
#     if(type(li)!=list):
#         return
#     if(len(li)==1):
#         return [li]
#     result=[]
#     for i in range(0,len(li[:])):
#         bak=li[:]
#         head=bak.pop(i) #head of the recursive-produced value
#         for j in Mideng(bak):
#             j.insert(0,head)
#             result.append(j)
#     return result

def CalcFitness(n, m, test_data):
    c_time1 = np.zeros([n, m])
    c_time1[0][0] = test_data[0][0]
    for i in range(1, n):
        c_time1[i][0] = c_time1[i - 1][0] + test_data[i][0]
    for i in range(1, m):
        c_time1[0][i] = c_time1[0][i - 1] + test_data[0][i]
    for i in range(1, n):
        for k in range(1, m):
            c_time1[i][k] = test_data[i][k] + max(c_time1[i - 1][k], c_time1[i][k - 1])
    return c_time1[n - 1][m - 1]

def NEH2(num_job, num_machine, test_data, num_factory):
    first_job = []
    factory_job_set = [[] for i in range(num_factory)]
    first_job_index = []
    factory_data = [[] for i in range(num_factory)]
    factory_fit = [[] for i in range(num_factory)]
    c_time = np.zeros([num_job, num_machine])
    #确定每一个工厂的第一个排序工件号
    temp = []
    k = 0
    for i in range(num_job):
        temp.append(sum(test_data[i]))
    first_job_index = np.argsort(temp)
    for i in range(num_factory):
        first_job.append(first_job_index[i] )
        factory_job_set[i].append(first_job[i])
        #为每个工厂分配第一个工件
        factory_data[i].append(test_data[first_job_index[i]])
    while True:
        if k == num_job - num_factory :
            break
        for i in range(num_factory):
            factory_job_set[i].append(first_job_index[num_factory + k])
            factory_data[i].append(test_data[first_job_index[num_factory + k]])
            factory_fit[i] = CalcFitness(len(factory_data[i]),num_machine,factory_data[i])
        #依次分配剩下的工件
        index = np.argsort(factory_fit)[0]
        for i in range(num_factory):
            if i == index:
                continue
            else:
                factory_job_set[i].pop()
                factory_data[i].pop()
        k += 1
    return  factory_job_set

def CalcFitness_sort(n, m, sort, test_data):
    c_time1 = np.zeros([n, m])
    c_time1[0][0] = test_data[sort[0]][0]
    for i in range(1, n):
        c_time1[i][0] = c_time1[i - 1][0] + test_data[sort[i]][0]
    for i in range(1, m):
        c_time1[0][i] = c_time1[0][i - 1] + test_data[0][i]
    for i in range(1, n):
        for k in range(1, m):
            c_time1[i][k] = test_data[sort[i]][k] + max(c_time1[i - 1][k], c_time1[i][k - 1])
    return c_time1[n-1][m-1]

def initial_Bayes(num_machine, num_factory, factory_job_set, test_data):
    #每个工厂的工件数
    len_job = [len(factory_job_set[i]) for i in range(num_factory)]
    #每个工厂的初始数据
    Mat_pop =[[[0 for i in range(len_job[k] + 1) ]for j in range(1000)] for k in range(num_factory)]
    for i in range(num_factory):
        for j in range(1000):
            sort = random.sample(factory_job_set[i], len_job[i])
            for k in range(len_job[i] + 1):
                if k == len_job[i]:
                    Mat_pop[i][j][k] = CalcFitness_sort(len_job[i], num_machine,
                                                        sort, test_data)
                else:
                    Mat_pop[i][j][k] = sort[k]
        Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-1])
        Mat_pop[i] = Mat_pop[i][0:200]
    return Mat_pop

def Bayes_update(Mat_pop, factory_job_set, num_factory):
    prob_mat_first = [[0.0 for i in range(len(factory_job_set[k]))] for k in range(num_factory)]
    #返回每个工厂相邻两个工件的所有情况的概率分布
    for i in range(num_factory):
        job_len = len(factory_job_set[i])
        #确定数据中第一个工件的出现概率
        index = 0
        demo = [ii[0] for ii in Mat_pop[i][0:100]]
        for job in factory_job_set[i]:
            prob_mat_first[i][index] = demo.count(job) / 100
            index += 1
        #每个工厂内分别有对应的（job_len - 1）个关系数组
    return prob_mat_first

def New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop):
    #每个工厂每次更新20个个体
    newpop = [[[-1 for i in range(len(factory_job_set[k]) + 2)] for j in range(100)] for k in range(num_factory)]
    #三维数组--第一维：工厂数；第二维：每个工厂的所有相邻关系数组；第三维：与上一个工件的关系
    prob_mat = [[[0.0 for i in range(len(factory_job_set[k]))]
                 for l in range(len(factory_job_set[k]) - 1)] for k in range(num_factory)]
    for i in range(num_factory):
        for num in range(100):
            temp = 0.0
            job_len = len(factory_job_set[i])
            #用轮盘赌确定第一个工件
            #temp_prob_mat = prob_mat[[[[:]:]:]:]
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

    return newpop

# def local_search(newpop, ls_frequency, factory_job_set, num_factory):
#     ls_pop = [[[[-1 for i in range(len(factory_job_set[k]))] for j in range(ls_frequency)] for l in range(20)]for k in range(num_factory)]
#     for i in range(num_factory):
#         for l in range(20):
#             ls_pop[i][l][0][:] = newpop[i][l][:]
#             for j in range(1, ls_frequency):
#                 temp1 = np.random.randint(0, len(factory_job_set[i]))
#                 temp2 = np.random.randint(0, len(factory_job_set[i]))
#                 while temp1 == temp2:
#                     temp1 = np.random.randint(0, len(factory_job_set[i]))
#                     temp2 = np.random.randint(0, len(factory_job_set[i]))
#                 temp = newpop[i][l][temp1]
#                 newpop[i][l][temp1] = newpop[i][l][temp2]
#                 newpop[i][l][temp2] = temp
#                 ls_pop[i][l][j][:-2] = newpop[i][l][:-2]
#                 newpop[i][l][:-2] = ls_pop[i][l][0][:-2]
#                 ls_pop[i][l][j][-2],ls_pop[i][l][j][-1] =
#     return ls_pop

def Bayes_net(pop_gen, ls_frequency):
    fitness = [[0 for i in range(ls_frequency)] for j in range(num_factory)]
    factory_job_set = NEH2(num_job, num_machine, test_data, num_factory)
    Mat_pop = initial_Bayes(num_machine, num_factory, factory_job_set, test_data)
    min_fitness = [0 for i in range(num_factory)]
    min_index = [0 for i in range(num_factory)]
    the_best = [[0 for i in range(pop_gen)] for j in range(num_factory)]
    the_worst = [[0 for i in range(pop_gen)] for j in range(num_factory)]
    temp_list = []
    data = [[] for i in range(num_factory)]
    for gen in range(pop_gen):
        prob_mat_first = Bayes_update(Mat_pop, factory_job_set, num_factory)
        newpop = New_pop(prob_mat_first, num_factory, factory_job_set, Mat_pop)
        ls_pop = local_search(newpop, ls_frequency, factory_job_set, num_factory)
        for i in range(num_factory):
            for j in range(ls_frequency):
                fitness[i][j] = CalcFitness_sort(len(factory_job_set[i]), num_machine, ls_pop[i][j], test_data)
            min_index[i] = np.argsort(fitness[i][:])[0]
            min_fitness[i] = fitness[i][min_index[i]]
        for i in range(num_factory):
            temp = len(factory_job_set[i])
            for j in range(200)[::-1]:
                if min_fitness[i] < float(Mat_pop[i][j][temp]):
                    temp_list = ls_pop[i][min_index[i]]
                    temp_list.append(min_fitness[i])
                    for l in range(len(factory_job_set[i]) + 1):
                        Mat_pop[i][j][l] = temp_list[l]
                    Mat_pop[i] = sorted(Mat_pop[i], key= lambda x:x[-1])
                    break
            the_best[i][gen] = Mat_pop[i][0][temp]
            the_worst[i][gen] = Mat_pop[i][-1][temp]
    return the_best,the_worst

