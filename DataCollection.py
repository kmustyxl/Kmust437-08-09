import xlwt
from AssignRule import *
book = xlwt.Workbook(encoding = 'utf-8')
sheet = book.add_sheet('data')
def demo():
    job = [10, 10, 10, 20, 20, 20, 30, 30, 30, 50]
    machine = [5, 10, 20, 5, 10, 20, 5, 10, 20, 5]
    for guimo in range(10):
        start_time = time.clock()
        num_machine = machine[guimo]
        num_job = job[guimo]
        test_data = ld.LoadData(num_job, num_machine)
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
            end_time = time.clock()
            if end_time - start_time <= 30:
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
            else:
                break
        sheet.write(guimo, 0, '%s_%s' % (num_job, num_machine))
        for i in range(num_factory):
            sheet.write(guimo, i*3 + 1, str(sum(the_best[i]) / gen))
            sheet.write(guimo, i*3 + 2, str(min(the_best[i][0:gen])))
            sheet.write(guimo, i*3 + 3, str(max(the_best[i][0:gen])))
           # sheet.write(guimo, i*3 + 4, )
    book.save('data\\data.xls')
