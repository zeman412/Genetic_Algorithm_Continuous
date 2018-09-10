# Implementation of Genetic Algorithm for solving a continuous optimization problem.
Min z  =  (x^2  +  y - 11)^2  +  (x +  y^2  - 7)^2

Where both the decision variables x and y lie between -6 and +6

As a benchmark the global minima is given:  Z = 0.0000,   x= 3.0000, y=2.0000
The problem is encoded as binary bit string and then applied single point crossover, bit flip mutation, 
and roulette wheel selection. As this is a minimization problem, the cumulative probability of each member 
in a population is calculated based on their distance from the minimum fitness value. After calculating 
the fitness of each member and subtracting the minimum fitness value, and then normalized it by dividing 
1 to the distance. Then these weight, which is based on how far it is from the minimum, is aggregated to 
a cumulative weight distribution. For the crossover, after deciding whether to do crossover based on the 
crossover probability, randomly picked a crossing point and did single point cross over. 
The mutation is done by flipping a bit based on the probability of mutation.
I used a chromosome length of 64-bit, finally I decoded this binary value and computed the fitness. 
The final chromosomes for the solution in binary look like these:
1100000000011101111111100011110010101010100100100011110011010001
1011111111110111000001001000010110101010101011100110001000101111
1100000000010101111011101101011110101010100110110110010111000111
