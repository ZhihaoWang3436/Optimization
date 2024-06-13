import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']


class AntTSP:

    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.visited = [False] * num_cities
        self.tour = []
        self.tour_length = 0

    def visit_city(self, city, distance_matrix):
        if self.tour:
            last_city = self.tour[-1]
            self.tour_length += distance_matrix[last_city, city]
        self.tour.append(city)
        self.visited[city] = True

    def tour_distance(self, distance_matrix):
        if len(self.tour) == self.num_cities:
            return self.tour_length + distance_matrix[self.tour[-1], self.tour[0]]
        return float('inf')

    def clear(self):
        self.visited = [False] * self.num_cities
        self.tour = []
        self.tour_length = 0


class AntColonyOptimizationTSP:

    def __init__(self, name, num_ants, num_iterations, alpha, beta, rho, q):
        self.name = name
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = None
        self.best_tour = None
        self.best_tour_length = float('inf')

    def initialize_pheromones(self, num_cities):
        self.pheromone = np.ones((num_cities, num_cities))

    def update_pheromones(self, ants, distance_matrix):
        self.pheromone = self.pheromone * (1 - self.rho)

        for ant in ants:
            contribution = self.q / ant.tour_distance(distance_matrix)
            for k in range(1, len(ant.tour)):
                i = ant.tour[k]
                j = ant.tour[k-1]
                self.pheromone[i][j] += contribution
                self.pheromone[j][i] += contribution

    def calculate_probability(self, current_city, next_city, distance_matrix):
        p1 = np.power(self.pheromone[current_city, next_city], self.alpha)
        p2 = np.power(distance_matrix[current_city, next_city], self.beta)
        return p1 / p2


    def select_next_city(self, ant, distance_matrix):
        current_city = ant.tour[-1]
        probabilities = np.zeros(ant.num_cities)
        for city in range(ant.num_cities):
            if not ant.visited[city]:
                probabilities[city] = self.calculate_probability(current_city, city, distance_matrix)
        total = probabilities.sum()
        probabilities = probabilities / total
        return np.random.choice(range(ant.num_cities), p=probabilities)

    def run(self, distance_matrix):
        num_cities = len(distance_matrix)
        self.initialize_pheromones(num_cities)
        ants = [AntTSP(num_cities) for _ in range(num_cities)]
        Best_tour_length = [0.] * self.num_iterations

        for i in range(self.num_iterations):
            for ant in ants:
                ant.clear()
                start_city = np.random.randint(0, num_cities-1)
                ant.visit_city(start_city, distance_matrix)
                while len(ant.tour) < num_cities:
                    next_city = self.select_next_city(ant, distance_matrix)
                    ant.visit_city(next_city, distance_matrix)
                if ant.tour_distance(distance_matrix) < self.best_tour_length:
                    self.best_tour = ant.tour
                    self.best_tour_length = ant.tour_distance(distance_matrix)

            self.update_pheromones(ants, distance_matrix)
            Best_tour_length[i] = self.best_tour_length
            print(f'第{i+1}次迭代的全局最优解如下:', self.best_tour_length)
        self.trace_plot(Best_tour_length)

        return self.best_tour, self.best_tour_length

    def trace_plot(self, Best_tour_length):
        plt.figure()
        plt.plot(np.arange(1, self.num_iterations + 1), Best_tour_length)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title(f'{self.name}')
        plt.savefig(f'rub\\{self.name}迭代过程.png')
        plt.show()



class AntValue:

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.position = np.zeros(num_dimensions)
        self.fitness = float('inf')

    def arrive(self, position):
        self.position = position

    def evaluate(self, objective_function):
        self.fitness = objective_function(self.position)


class AntColonyOptimizationValue:

    def __init__(self, name, num_ants, num_iterations, alpha, beta, rho, q, num_dimensions, bounds):
        self.name = name
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.pheromone = None
        self.best_position = None
        self.best_fitness = float('inf')

    def initialize_pheromones(self):
        # self.pheromone = np.ones((self.num_dimensions, 2))
        self.pheromone = np.random.random((self.num_dimensions, 2))

    def update_pheromones(self, ants):
        self.pheromone *= (1 - self.rho)
        for ant in ants:
            contribution = self.q / ant.fitness
            for i in range(self.num_dimensions):
                lower_bound, upper_bound = self.bounds[i]
                normalized_position = (ant.position[i] - lower_bound) / (upper_bound - lower_bound)
                self.pheromone[i, 0] += contribution * (1 - normalized_position)
                self.pheromone[i, 1] += contribution * normalized_position

    def select_next_position(self):
        new_position = np.zeros(self.num_dimensions)
        for i in range(self.num_dimensions):
            lower_bound, upper_bound = self.bounds[i]
            prob = np.power(self.pheromone[i], self.alpha)
            prob /= prob.sum()
            r = np.random.rand() / 2  # 增加随机性
            normalized_position = np.inner(prob, [r, 1-r])
            new_position[i] = lower_bound + normalized_position * (upper_bound - lower_bound)
        return new_position

    def run(self, objective_function):
        ants = [AntValue(self.num_dimensions) for _ in range(self.num_ants)]
        self.initialize_pheromones()
        Best_fitness = np.zeros(self.num_iterations)

        for i in range(self.num_iterations):
            for ant in ants:
                position = self.select_next_position()
                ant.arrive(position)
                ant.evaluate(objective_function)
                if ant.fitness < self.best_fitness:
                    self.best_fitness = ant.fitness
                    self.best_position = ant.position
            self.update_pheromones(ants)
            Best_fitness[i] = self.best_fitness
            # print(f'第{i + 1}次迭代的全局最优点如下:', self.best_position)
            print(f'第{i + 1}次迭代的全局最优解如下:', self.best_fitness)

        self.trace_plot(Best_fitness)

        return self.best_position, self.best_fitness

    def trace_plot(self, Best_fitness):
        plt.figure()
        plt.plot(np.arange(1, self.num_iterations + 1), Best_fitness)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title(f'{self.name}')
        plt.savefig(f'rub\\{self.name}迭代过程.png')
        plt.show()