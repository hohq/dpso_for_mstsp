import numpy as np
import math
mstsp_pr = ["simple1_9", "simple2_10", "simple3_10", "simple4_11", "simple5_12", "simple6_12",
                    "geometry1_10", "geometry2_12", "geometry3_10", "geometry4_10", "geometry5_10", "geometry6_15",
                    "composite1_28", "composite2_34", "composite3_22", "composite4_33", "composite5_35", "composite6_39",
                    "composite7_42", "composite8_45", "composite9_48", "composite10_55", "composite11_59",
                    "composite12_60", "composite13_66"]

solution_folder="C:/software/programming/mstsp/benchmark_MSTSP/"
algorithm_folder="C:/software/programming/mstsp/results/"

def create_similarity_matrix(vec):
    len_vec = len(vec)
    similarity_matrix = np.zeros((len_vec, len_vec))
    for i in range(len_vec):
        for j in range(len_vec):
            if vec[i] == j :
                
                # print("j")
                # print(j)
                # print("vec[j]")
                # print(vec[j])
                # print("i")
                # print(i)
                similarity_matrix[i, j] = 1
    return similarity_matrix

def measure_share_dist(vec1, vec2):
    # 创建两个解决方案的相似矩阵
    min_len = min(len(vec1), len(vec2))
    
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    # print("vec1")
    # print(vec1)
    # print("vec2")
    # print(vec2)
    
    similarity_matrix_1 = create_similarity_matrix(vec1)
    similarity_matrix_2 = create_similarity_matrix(vec2)
    # print("similarity_matrix_1")
    # print(similarity_matrix_1)
    # print("similarity_matrix_2")
    # print(similarity_matrix_2)
    
    # 计算共享距离
    share_dist = sum(sum(np.logical_and(similarity_matrix_1, similarity_matrix_2)))
    # print("share_dist")
    # print(share_dist)
    return share_dist

# 示例
# vec1 = [1, 2, 3, 4, 5]
# vec2 = [5, 4, 3, 2, 1]
# print(measure_share_dist(vec1, vec2))



# def read_problem(index):
#     problem_file= mstsp_pr[index]+".tsp"
#     return problem_file


# def read_solution(index):
#     solution_file= mstsp_pr[index]+".solution"
#     with open(solution_folder+"/"+solution_file) as f:
#         lines = f.readlines()
        
        

# def read_algorithm(file_name):
#     with open(algorithm_folder+"/"+file_name) as f:
#         lines = f.readlines()
#         n = int(lines[0])
#         m = int(lines[1])
#         algorithm = np.zeros((n,m),dtype=int)
#         for i in range(2,n+2):
#             line = lines[i].split()
#             for j in range(m):
#                 algorithm[i-2,j] = int(line[j])
#         return algorithm
    
    
def measure(index):
    # problem= read_problem(index)
    # solution = read_solution("solution"+str(index)+"_1")
    solution_file= solution_folder+mstsp_pr[index]+".solution"
    mstsp_solution = np.loadtxt(solution_file).astype(np.int64)   # 修改这里
    # mstsp_solution = mstsp_solution[:, :-1]
    algorithm_file = f"C:/software/programming/mstsp/results/{mstsp_pr[index]}.alg_solution"
    algorithm_result = np.loadtxt(algorithm_file).astype(np.int64)   # 修改这里
    j=0
    i=0
    share_dist = np.zeros((len(mstsp_solution), len(algorithm_result)))  # 初始化为零的二维数组
    algorithm_distance_list = []
    for algorithm_line in algorithm_result:
        algorithm_distance=algorithm_line[0]
        algorithm_distance_list.append(algorithm_distance)
        
    for solution_line in mstsp_solution:
        # solution_distance = solution_line[0]
        # print("solution_distance")
        # print(solution_distance)
        solution_vec = solution_line[1:-1]
        # print("solution_vec")
        # print(solution_vec)
        j=0
        for algorithm_line in algorithm_result:
            # algorithm_distance=algorithm_line[0]
            # algorithm_distance_list.append(algorithm_distance)
            # print("algorithm_distance")
            # print(algorithm_distance)
            algorithm_vec = algorithm_line[1:]
            # print("algorithm_vec")
            # print(algorithm_vec)
            share_dist[i][j]=measure_share_dist(algorithm_vec,solution_vec)
            j+=1
        i+=1
    # print("algorithm_distance_list")
    # print(algorithm_distance_list)
    average_distance = np.mean(algorithm_distance_list)

    std_deviation = np.std(algorithm_distance_list)

    
    
    # print("share_dist")
    # print(share_dist)
    
    city_num = len(solution_vec)
    # print("city_num")
    # print(city_num)
    # flag_mstsp=[]
    flag_mstsp = np.max(share_dist, axis=1) == np.tile(city_num, (mstsp_solution.shape[0], 1))
    # for i in share_dist:
    #     flag_mstsp.append(np.max(i) == city_num)
    
    # print("flag_mstsp")
    # print(flag_mstsp)
    
    num_columns = share_dist.shape[1]  # 获取列的数量
    flag_alg=[]
    
    flag_alg = np.max(share_dist, axis=0) == np.tile(city_num, (algorithm_result.shape[0], 1)).T
    # for j in range(num_columns):
        # flag_alg.append(np.max(share_dist[:, j]) == city_num)   
            
    # print("flag_alg")
    # print(flag_alg)
    near_num = np.max(share_dist, axis=0)
    # print("near_num")
    # print(near_num)
    
    
    beta2 = 0.3;
    DI=np.average(near_num)/len(solution_vec)
    TP = np.sum(flag_alg)
    FP = flag_alg.shape[0] - TP
    FN = flag_mstsp.shape[0] - np.sum(flag_mstsp)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    Fbeta = (1 + beta2) * P * R / ((beta2) * P + R)
    
    print("problem: ", mstsp_pr[index])
    print("DI: ", DI)
    print("Fbeta: ", Fbeta)
    print("平均距离:", average_distance)
    print("标准差:", std_deviation)
    
       
if __name__=="__main__":
    for i in range(24):
        measure(i)
        

    