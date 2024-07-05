from builtins import object
import numpy as np
import math
import os
import random
import time
dist = np.array([])
MaxFES = np.array([])
FES = 0
object_input = 0
MaxFES = np.append(MaxFES, [60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000,
                           1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000,
                           1200000, 1200000, 1200000, 1200000, 1200000, 1200000])
mstsp_pr = ["simple1_9", "simple2_10", "simple3_10", "simple4_11", "simple5_12", "simple6_12",
                    "geometry1_10", "geometry2_12", "geometry3_10", "geometry4_10", "geometry5_10", "geometry6_15",
                    "composite1_28", "composite2_34", "composite3_22", "composite4_33", "composite5_35", "composite6_39",
                    "composite7_42", "composite8_45", "composite9_48", "composite10_55", "composite11_59",
                    "composite12_60", "composite13_66"]


class cardinate:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

class City(object):
    city_vec = np.array([])
    random = np.array([])
    onpath = np.array([])
    fitness = 0
    path_length = 0
    d = cardinate()

    def distance(self, a, b):
        return round(math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))

    def __init__(self, input_data):
        global object_input
        object_input = input_data
        self.random = np.array([])
        mstsp_ds = ["simple1_9", "simple2_10", "simple3_10", "simple4_11", "simple5_12", "simple6_12",
                    "geometry1_10", "geometry2_12", "geometry3_10", "geometry4_10", "geometry5_10", "geometry6_15",
                    "composite1_28", "composite2_34", "composite3_22", "composite4_33", "composite5_35", "composite6_39",
                    "composite7_42", "composite8_45", "composite9_48", "composite10_55", "composite11_59",
                    "composite12_60", "composite13_66"]

        filename = "benchmark_MSTSP/" + str(mstsp_ds[input_data]) + ".tsp"
        file_ = open(filename, "r")

        # Reading from the file
        content = file_.readlines()

        # Iterating through the content
        # Of the file
        for line in content:
            x, y = list(map(int, line.split()))
            self.city_vec = np.append(self.city_vec, cardinate(x, y))

        file_.close()
        city_num = len(self.city_vec)
        self.city_num = city_num
        global dist

        # filename2 = "benchmark_MSTSP/" + str(mstsp_ds[input_data]) + ".solution"
        # file_2 = open(filename2, "r")

        # content = file_2.readlines()

        # global minf
        # minf = 0
        # global sol
        # sol = []
        # for line in content:
        #     sol1 = list(map(int, line.split()))
        #     sol1.pop()
        #     minf = sol1.pop(0)
        #     sol.append(sol1)

        dist = np.empty((city_num, city_num), dtype=float)

        for i in range(city_num):
            for j in range(city_num):
                dist[j][i]=dist[i][j] = self.distance(self.city_vec[i], self.city_vec[j])

    def tot_dist(self, new_vec):
        global MaxFES, object_input, FES, dist
        if FES >= MaxFES[object_input]:
            return
        new_city_vec = np.array([])
        self.path_length = 0
        self.fitness = 0
        for i in range(self.city_num - 1):
            self.path_length += dist[int(new_vec[i][1])][int(new_vec[i + 1][1])]

        self.path_length += dist[int(new_vec[self.city_num - 1][1])][int(new_vec[0][1])]
        self.fitness = self.path_length
        FES += 1

    def _get_maxfes(self, ml):
        object_input = ml
        return MaxFES[object_input]

    # def _evaluate(self, arr):
    #     arr1 = np.array([])
    #     arr1 = np.append(arr1, arr)
    #     tempRandom = Chromosome(arr)
    #     self.random = tempRandom.path
    #     self.onpath = tempRandom.op
    #     self.tot_dist(self.random)
    #     return self.fitness, self.onpath

    # def _for_sol(self):
    #     return sol

    # def _min_fit(self):
    #     return minf


class Particle:
    def __init__(self, city):
        # 初始化位置为城市的一个排列
        self.position = np.random.permutation(city.city_num)
        # 初始化速度为一系列替换操作，这里简化为随机选择两个城市进行替换
        # 假设初始速度包含一定数量的替换操作
        self.velocity = [(np.random.randint(city.city_num), np.random.randint(city.city_num)) for _ in range(city.city_num)]
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness=self.evaluate(dist)
        

    def evaluate(self, city_dist_matrix):
        global FES
        FES+=1;
        # print(FES);
        total_distance = 0
        for i in range(len(self.position) - 1):
            total_distance += city_dist_matrix[self.position[i]][self.position[i+1]]
        total_distance += city_dist_matrix[self.position[-1]][self.position[0]]  # Return to start
        return total_distance

        
    def update_velocity(self, global_best_position,current_best_position,w, c1, c2):
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        new_velocity=[]
        for op in self.velocity:
            if random.random() < w:  # 使用概率w决定是否保留
                new_velocity.append(op)

        if random.random() < c1 * r1:
            velocity_update=self.calculate_velocity(self.best_position, self.position)
            if isinstance(velocity_update, list):
                new_velocity.extend(velocity_update)  # 添加列表中的每个元素
            else:
                new_velocity.append(velocity_update)
                
        if random.random() < c2 * r2:
            velocity_update = self.calculate_velocity(global_best_position, self.position)
            if isinstance(velocity_update, list):
                new_velocity.extend(velocity_update)  # 添加列表中的每个元素
            else:
                new_velocity.append(velocity_update)
                
        self.velocity=new_velocity

    
    def calculate_velocity(self,position1, position2):
    #两个位置得到新的速度
        velocity = []
        for i, city in enumerate(position1):
            if city != position2[i]:
                # 找到city在position2中的位置
                target_index = np.where(position2 == city)[0][0]
                # 记录替换操作：(当前位置, 目标位置)
                velocity.append((i, target_index))
        return velocity
   
    # best_solutions = []
    def update_position(self):
        # self.position = np.clip(self.position + self.velocity, 0, 1)
        
        # print("before")

        # print(self.velocity)
        # print(self.position)
        for swap in self.velocity:
        # 使用np.where找到swap[0]和swap[1]在self.position中的索引
            if not swap:
                continue  # 如果swap为空，则跳过当前迭代
            # print("swap")
            # print(swap)
            # print(swap[0],swap[1])

            index1 = np.where(self.position == swap[0])[0]
            index2 = np.where(self.position == swap[1])[0]
           
            # print("done")            
            # 交换位置
            self.position[index1], self.position[index2] = self.position[index2], self.position[index1]
        # print("after")
        # print(self.position)
            
        fitness = self.evaluate(dist)
        if fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness



class PSO:
    def __init__(self, city, num_particles, max_iterations, w, c1, c2):
        self.city = city
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.best_solutions = []

    # 更新最优解列表的函数
    
            
    def solve(self):
        particles = [Particle(self.city) for _ in range(self.num_particles)]
        global_best_position = particles[0].best_position.copy()
        current_best_position = particles[0].best_position.copy()        
        global_best_fitness = particles[0].best_fitness

        for particle in particles:
            if particle.best_fitness < global_best_fitness:
                global_best_position = particle.best_position.copy()
                global_best_fitness = particle.best_fitness
        w_start=self.w
        w_end=0.1
        # c1_start = self.c1
        # c1_end = 1.0
        # c2_start = self.c2  # 增加c2的初始值
        # c2_end = 1.0  # 在迭代过程中逐渐减小c2以增强局部搜索能力

        for now_iterations in range(self.max_iterations):
            w = w_start - ((w_start - w_end) / self.max_iterations) * now_iterations
            # c1 = c1_start - ((c1_start - c1_end) / self.max_iterations) * now_iterations  # 动态调整c1
            # c2 = c2_start - ((c2_start - c2_end) / self.max_iterations) * now_iterations  # 动态调整c2
            c1=self.c1
            c2=self.c2
            for particle in particles:
                particle.update_velocity(global_best_position,current_best_position, w, c1, c2)
                particle.update_position()
                
                if particle.best_fitness < global_best_fitness:
                    global_best_position = particle.best_position.copy()
                    global_best_fitness = particle.best_fitness
                    self.best_solutions=self.update_pso_best_solutions(global_best_fitness,global_best_position)
            # print(global_best_position,global_best_fitness)

        return global_best_position, global_best_fitness

    def update_pso_best_solutions(self,fitness, position):
        self.best_solutions.append({'fitness': fitness, 'position': position})
        # 按照fitness值的升序排列列表
        self.best_solutions.sort(key=lambda x: x['fitness'])
        # 如果列表长度超过10，移除fitness值最大的解
        if len(self.best_solutions) > 5:
            self.best_solutions.pop()
        return self.best_solutions





def update_best_solutions(fitness, position,best_solutions):
        # 将新解添加到列表中
        best_solutions.append({'fitness': fitness, 'position': position})
        # 按照fitness值的升序排列列表
        best_solutions.sort(key=lambda x: x['fitness'])
        # 如果列表长度超过10，移除fitness值最大的解
        if len(best_solutions) > 15:
            best_solutions.pop()
        return best_solutions
    
if __name__ == "__main__":
    start_time = time.time()

    
    output_dir = "C:/software/programming/mstsp/results/"
    run=1
    if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    for i in range(24):
        problem_name=mstsp_pr[i]
        FES=0
        best_solutions=[]
        for j in range(50):
            run=j
            
            
            city = City(i)  # Replace 0 with the desired input_data index
            pso = PSO(city, num_particles=203, max_iterations=100, w=0.618, c1=2.5324, c2=1.0086)
            best_position, best_fitness = pso.solve()
            for best in pso.best_solutions:
                best_fitness = best['fitness']
                best_position = best['position']
                
                best_solutions=update_best_solutions(best_fitness, best_position,best_solutions)
            if(j==0): 
                fes_one=FES
                # print(fes_one)
            if (FES+fes_one>MaxFES[i]+500):
                break
        
        file_string = f"{output_dir}{problem_name}.alg_solution"
        with open(file_string, 'w') as ofile_record:
                for best in best_solutions:
                    ofile_record.write(f"{best['fitness']}\t")
                    for k in range(city.city_num):
                        ofile_record.write(f"{best['position'][k]}\t")                
                
                    ofile_record.write("\n")
                    # ofile_record.write(f"Best Fitness: {best_fitness}\n")  # 输出最佳适应度值
            # print("Best Position:", best_position)
            # print("Best Fitness:", best_fitness)
        print(f"done{problem_name}")
            
        # print("fes:",FES)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

