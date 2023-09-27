import numpy
import matplotlib.pyplot as plt
import math
import random




# Helper function to xor two characters
def xor_c(a, b):
    return '0' if(a == b) else '1';
 
# Helper function to flip the bit
def flip(c):
    return '1' if(c == '0') else '0';

def graytoBinary(gray):
 
    binary = "";
 
    # MSB of binary code is same
    # as gray code
    binary += gray[0];
 
    # Compute remaining bits
    for i in range(1, len(gray)):
         
        # If current bit is 0,
        # concatenate previous bit
        if (gray[i] == '0'):
            binary += binary[i - 1];
 
        # Else, concatenate invert
        # of previous bit
        else:
            binary += flip(binary[i - 1]);
 
    return binary;

# A method to calculate the first objective function
def first_objective_function(x):
    x = convert_to_binary(x)
    binary = graytoBinary(x)
    integer = int(binary, 2)
    return math.sin((integer * math.pi)/256)

# A method to calculate the second objective function
def second_objective_function(x,y):
    result = pow((x - 3.14),2) + pow((y - 2.72),2) + math.sin(3*x + 1.41) + math.sin( 4*y - 1.73)
    return result

# A method that generates offspring using randomly generated one point crossover
def single_point_crossover(A,B,x):
    A_new = numpy.append(A[:x], B[x:])
    B_new = numpy.append(B[:x], A[x:])
    return A_new, B_new

# A method that generates offspring using crossover point at the middle
def fifty_percent_crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.

    crossover_point = numpy.uint8(offspring_size[1]/2)
    for k in range(0, offspring_size[0], 2):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        # Two offspring using sinlge point crossover
        new_children = single_point_crossover(parents[parent1_idx], parents[parent2_idx], crossover_point)
        
        #adding the new offspring
        offspring[k] = new_children[0]
        offspring[k+1] = new_children[1]
        
    return offspring

# A method that generates offspring using multicrossover
def multi_point_crossover(A,B,x1,x2):
    A,B = single_point_crossover(A,B,x1)
    A,B = single_point_crossover(A,B,x2)  
    return A,B

# A method that returns the new generated offspring from the one point crossover
def one_point_crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. It is randomized
    crossover_point = random.randint(1,offspring_size[1]-1)
    
    for k in range(0,offspring_size[0],2):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx,0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx,crossover_point:]
    return offspring


# A method that returns the new generated offspring from the two point crossover
def two_point_crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    """
    Todo : make one point crossover random
    """
    crossover_first_point = random.randint(0, offspring_size[1]-1)
    crossover_second_point = random.randint(1, offspring_size[1]-1)
    
    while(crossover_second_point >= crossover_first_point):
        crossover_first_point = random.randint(0, offspring_size[1]-1)
        crossover_second_point = random.randint(1, offspring_size[1]-1)
    for k in range(0,offspring_size[0],2):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        new_children = multi_point_crossover(parents[parent1_idx], parents[parent2_idx], crossover_first_point, crossover_second_point)
        offspring[k] = new_children[0]
        offspring[k+1] = new_children[1]
    return offspring


# A method that does mutation for integer values with prob
def mutation_integer(population, prob):
    for i in range(len(population)):
        for j in range(len(population[0])):
            if (random.random() < prob):
                population[i][j]= random.uniform(-5, 5) #generate a random number from -5 to 5
    return population

# A method that does mutation for binary values with prob
def mutation_binary(population, prob):
   for i in range(len(population)):
       for j in range(len(population[0])):
            if random.random() < prob:
                population[i][j] = 0 if population[i][j] else 1 #generate a random binary number 0 or 1

   return population

def convert_binary_to_array(binary):
    return binary.toCharArray()

def convert_array_to_binary(arr):
    string= ''
    # traverse in the string
    for ele in arr:
        string += str(ele)
    return string

def convert_to_binary(arr):
    string= ''
    # traverse in the string
    for ele in arr:
        x=int(ele)
        string += str(x)
    return string

# A method that generates intial integer population with a size
def generate_intial_integer_pop(pop_size):
    pop = numpy.zeros(shape=(pop_size, 2))
    for i in range(pop_size):
        for j in range(2):
            pop[i][j] = random.uniform(-5, 5)
    return pop

# A method that generates intial binary population with a size
def generate_initial_binary_pop(pop_size):
    pop = numpy.zeros(shape=(pop_size, 8))
    for i in range(pop_size):
        for j in range(8):
            x=random.randint(0, 1)
            pop[i][j] = x   
    return pop

# A method that returns the fitness of the given population
def cal_pop_fitness(pop, objective):
    fitness = numpy.empty(len(pop))
    for i in range(len(pop)):
        # choose which objective function to use according to the variable 'objective'
        if(objective == 1):
            fitness[i] = first_objective_function(pop[i])
        else:
            #it is assigned - the objective function because we want to get the global minimum
            #so it should be maximum the fitness function
            fitness[i] = - second_objective_function(pop[i][0],pop[i][1])
    return fitness

# A method that chooses half the population in order to continue to the next population 
# Chooses the best fitnesses to continue to the next population
def select_mating_pool(pop, fitness):
    for i in range(int(len(pop)/2)):
        minIndex = fitness.argmin()
        fitness = numpy.delete(fitness, minIndex)
        ## fitness[i] = fitness.delete(maxIndex)
        pop = numpy.delete(pop, minIndex, axis = 0)
    return pop

def binaryToDecimal(n):
    return int(n,2) 

# A repair funtion for integers
def repair(pop):
    min_value = -5
    max_value = 5
    for i in range(len(pop)):
        if(pop[i][0] > max_value):
            pop[i][0] = 5
        if(pop[i][1] > max_value):
            pop[i][1] = 5
        if(pop[i][0] < min_value):
            pop[i][0] = -5
        if(pop[i][1] < min_value):
            pop[i][1] = -5
    return(pop)

# A repair funtion for binary
def repair_1(pop):
    min_value = 0
    max_value = 255
    
    for i in range(len(pop)):
        x=convert_to_binary(pop[i])
        y=binaryToDecimal(x)
        if y>max_value:
            pop[i] = max_value
        if y<min_value:
            pop[i] = min_value
    return(pop)

#print(multi_point_crossover([1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1], 2,5))
#rint(convert_array_to_binary([1,1,1,1,1,0,0,0]))  

####
pop_size = 16
generations = range(0,20)




#########################################
## Genetic Algorithms for the second objective function (integer)

best_fitness = []
avg_fitness = []
pop = generate_intial_integer_pop(pop_size)
for i in generations: 
    print("generation ", i ,": ")
    print()
    pop = repair(pop)
    print(pop)
    print()
    fitness = cal_pop_fitness(pop, 2)
    print(fitness)
    print()
    best_fitness.append(numpy.max(fitness))
    avg_fitness.append(numpy.average(fitness))
    selected_parents = select_mating_pool(pop, fitness)
    print(selected_parents)
    print()
    
    offspring = fifty_percent_crossover(selected_parents, [int(pop_size/2),2])
    pop = numpy.concatenate((selected_parents, offspring), axis=0)
    print(offspring)
    print()
    pop = mutation_integer(pop, 0.02)
    print(pop)
    print()
    


plt.title("Best Fitnesses for the second objective functions")
plt.plot(generations, avg_fitness, color="red")

plt.show()

## Genetic Algorithms for the first objective function (binary)
best_fitness_binary = []
avg_fitness_binary = []
pop = generate_initial_binary_pop(pop_size)
for i in generations:
    print("generation ", i ,": ")
    print()
    pop = repair_1(pop)
    print("POP is")
    print(pop)
    print()
    fitness = cal_pop_fitness(pop, 1)
    print("Fitness is")
    print(fitness)
    print()
    x=random.randint(0, 1)
    best_fitness_binary.append(numpy.max(fitness))
    avg_fitness_binary.append(numpy.average(fitness))
    selected_parents = select_mating_pool(pop, fitness)
    print("selected is")
    print(selected_parents)
    print()
       
    if x==0:
            offspring = two_point_crossover(selected_parents, [int(pop_size/2),8])
    else:    
            offspring = one_point_crossover(selected_parents, [int(pop_size/2),8])
                 
    pop = numpy.concatenate((selected_parents, offspring), axis=0)
    print()
    print("Offspring is")
    print(offspring)
    print()
    print("Mutated is")
    pop = mutation_binary(pop, 0.01)
    print(pop)
    print()

plt.title("Average Fitnesses for the first objective function")
plt.plot(generations, best_fitness_binary, color="red")
plt.show()