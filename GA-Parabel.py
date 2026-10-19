import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

"""
Die Gleichung  y=ax2+bx+c ergibt, wenn man sie grafisch darstellt, eine Parabel. 
Ziel des genetischen Algorithmus (GA) ist es eine Kombination der Werte für 
a, b und c zu finden, die die flachste Parabel ergibt.

Es gibt zwei implementierte Abbruchbedigungen zwischen denen man wählen kann:
1) Bis eine fest definierte Anzahl an Generationen erreicht ist.
2) Bis ein Fitnesswert erreicht ist.
"""

best_fitness_history: list[float] = []
worst_fitness_history: list[float] = []
genes_of_best_individuals: list[tuple[float, float, float]] = []

# Bewertet wie flach eine nach oben geöffnete Parabel im Bereich x=-1 bis x=1 ist.
# Je flacher desto besser.
# Dieser Krümmungswert wird zurückgegeben.
def fitness_function_orig(params) -> float:
    a, b, c = params
    if a <= 0: # Geraden & nach unten geöffnete Parabeln werden verworfen, indem sie bestraft werden.
        return -float('inf')  # Bestrafung mit -unendlich
    vertex_x = -b / (2 * a) # x-Wert des Scheitelpunktes
    vertex_y = a * (vertex_x ** 2) + b * vertex_x + c # y-Wert des Scheitelpunktes
    y_left = a * (-1) ** 2 + b * (-1) + c #y-Wert bei x = -1
    y_right = a * (1) ** 2 + b * (1) + c #y-Wert bei x = 1
    curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y) # Höhenunterschied zwischen Rändern und Scheitelpunkt
    # Flache Parabel hat an den Rändern kaum Höhenunterschied zum SP
    # 0.0000000000000001

    # Krümmung negieren, denn Algo sucht nach maximaler Fitness:
    #   curviness = 50 -> Fitness = -50 (schlecht)
    #   curviness = 0,01 -> Fitness = -0,01 (gut)
    return -curviness


# korrigiertes Original
def fitness_function_original(params) -> float:
    a, b, c = params
    if a <= 0: # Geraden & nach unten geöffnete Parabeln werden verworfen, indem sie bestraft werden.
        return -float('inf')  # Bestrafung mit -unendlich
    vertex_x = -b / (2 * a) # x-Wert des Scheitelpunktes
    vertex_y = a * (vertex_x ** 2) + b * vertex_x + c # y-Wert des Scheitelpunktes
    y_left = a * (vertex_x-1) ** 2 + b * (vertex_x-1) + c #y-Wert bei SP-1
    y_right = a * (vertex_x+1) ** 2 + b * (vertex_x+1) + c #y-Wert bei SP+1
    curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y) # Höhenunterschied zwischen Rändern und Scheitelpunkt
    # Flache Parabel hat an den Rändern kaum Höhenunterschied zum SP
    # 0.0000000000000001

    # Krümmung negieren, denn Algo sucht nach maximaler Fitness:
    #   curviness = 50 -> Fitness = -50 (schlecht)
    #   curviness = 0,01 -> Fitness = -0,01 (gut)
    return -curviness


def fitness_function(params) -> float:
    a, b, c = params
    if a < 0:
        return -float('inf')
    x_vertex: float = -b / (2*a) # x-Wert des Scheitelpunktes
    first_derivative_x: float = 2*a*x_vertex+b # x-Wert des SP eingesetzt in erste Ableitung
    second_derivative_x: float = 2*a # x-Wert des SP eingesetzt in zweite Ableitung
    kappa: float = abs(second_derivative_x / (1+(first_derivative_x)**2)**(3/2)) # Kürmmungsstärke im Scheitelpunkt
    #r = 1/abs(kappa) # Radius der Krümmung im Scheitelpunkt
    return -kappa


# Erzeugt Start-Population
# Beispiel-return: [(17.594801325753266, 10.300852625563174, -10.857726749407185), (...), ...]
# Eine Population besteht aus Individuen.
# Ein Individuum besteht aus dem Triplett a, b, c
def create_initial_population(size, lower_bound, upper_bound) -> list[tuple[float, float, float]]:
    population: list[tuple[float, float, float]] = []
    for _ in range(size): # '_' wird verwendet, wenn man Schleifenzähler nicht braucht.
        individual: tuple[float, float, float] = (random.uniform(lower_bound, upper_bound),
                                                  random.uniform(lower_bound, upper_bound),
                                                  random.uniform(lower_bound, upper_bound))
        population.append(individual)
    return population


# Selection function using tournament selection
# Ermittelt Elternpopultion
# Dabei werden (wenn nicht anders aufgerufen mit 'tournament_size') immer die Fitnesswerte von 
# DREI Individuen miteinander vergleichen. Das stärkste Individuum wird in 'selected' hinzugefügt.
# Die Eltern-Population 'selected' bleibt genau so groß, wie die Ausgangspopulation 'population', nämlich '100'.
# Das heißt aber nicht, dass jedes Inidviduum ein Elternteil wird. Denn im Turnament setzt sich immer das stärkste durch.
# Der stärkste kann sich mehrmals durchsetzen und ein Turnier somit mehrmals gewinnen.
# Die Listengröße reduziert sich also nicht von 100 auf x, sondern ist eine gleichgroße Kopie aus den besten Individuen der alten Liste.
# return-Beispiel: [(40.31985375805485, 9.199362397617548, -7.156103274302907), (18.849825421295037, 8.019170588836111, -2.620513681422821), ...]
def selection(population, fitnesses, tournament_size=3) -> list[tuple[float, float, float]]:
    selected: list[tuple[float, float, float]] = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        # tournament:
        # [
        #  ((49.76589218061807, -3.9896551803142657, 27.649186912150554), -99.69170662687179),
        #  ((-6.174565314838645, -15.435534160030869, -41.217882925845394), -inf),
        #  ((11.524711885400905, -3.805915495148369, 34.368532736128785), -23.677855652667745)
        # ]
        # zip(population, fitness) -> [((a1, b1, c1), fitness1), ((a2, b2, c2), fitness2), ...]
        # random.sample(liste, 3) -> Ziehen von drei zufälligen Paaren für Turnier, ohne Zurücklegen.
        winner = max(tournament, key=lambda x: x[1])[0] # (11.524711885400905, -3.805915495148369, 34.368532736128785)
        # x ist Individuum
        # x[1] ist Fitnesswert von Individuum
        selected.append(winner)
    return selected


# Die Tripletts zweier Elternindividuen werden zufällig rekombiniert zu zwei neuen Kindindividuen.
# Gleicher Mishfaktor "Alpha" sorgt für gleichmäßige Verteilung der Gene an Kind.
# Spiegelverkehrte Formel verhindert zwei identische Kindklone.
# Genau zwei Kinder als Rückgabewert ist durch paarweises Durchlaufen der Schleife bedingt.
# Ein Kind würde jede Generation halbieren. Mehr als zwei Kinder würde Generationen explodieren.
def crossover(parent1, parent2) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    alpha: float = random.random() # Zufallszahl
    child1 = tuple(alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2))
    child2 = tuple(alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2))
    return child1, child2

# Mutiert die Gene eines Individuum/Chromosoms
def mutation(individual, mutation_rate, lower_bound, upper_bound) -> tuple[float, float, float]:
    individual = list(individual) # Tupel vonvertieren zu Liste - immutale zu mutable
    for i in range(len(individual)): # Jedes Gen iterieren
        if random.random() < mutation_rate: # Bestimmt, ob eine Mutation stattfindet. random.random() erzeugt Zufallszahl im Intervall [0.0; 1.0)
            #mutation_amount = random.uniform(-1, 1) # Bestimmt, wie stark mutiert wird. Jeder Wert zwischen -1 und 1 kann mit der gleichen Wahrsch. drankommen.
            mutation_amount = random.gauss(0, 0.01) # Bestimmt, wie stark mutiert wird. Die Werte sind normalverteilt.
            individual[i] += mutation_amount
            # Ensure the individual stays within bounds
            individual[i] = max(min(individual[i], upper_bound), lower_bound) # Gen-Wert innerhalb der Suchgrenzen (upper_bound & lower_bound) halten
    return tuple(individual)

# Berechnet den x-Wert des Scheitelpunktes eines Individuum anhand seiner Gene.
def get_x_vertix(genes: tuple[float, float, float]) -> float:
    a, b, _ = genes
    x_vertex: float = -b / (2*a)
    return x_vertex


# Main genetic algorithm function
# Ermittelt das Individuum mit der höchsten Fitness der letzten Generation
# Rückgabe inkludiert:
#   Generation des besten Individuums
#   Gene des besten Individuums -> a, b, c
#   Fitnesswert des besten Individuums
#   x-Scheitelpunktwert des besten Individuums
def genetic_algorithm(population_size: int, mutation_rate: float, lower_bound: int, upper_bound: int, target_fitness: float = 0.0, max_generation: int = 20000, stagnation_limit: int = 15) -> tuple[int, float, float, float, float]:
    current_fitness = float('inf')
    current_geneation: int = 0
    global best_fitness_history # Nur fürs plotten später relevant
    global worst_fitness_history # Nur fürs plotten später relevant
    global genes_of_best_individuals # Nur fürs plotten später relevant
    
    population: list[tuple[float, float, float]] = create_initial_population(population_size, lower_bound, upper_bound) # Ausgangspopulation

    while current_fitness != target_fitness:
        
        if current_geneation == max_generation:
            print("ABBRUCH - Maximale Anzahl an Generationen erreicht!\n")
            break
        # Prüft, ob die letzten `stagnation_limit` Werte alle nahe an `current_fitness` liegen
        elif len(best_fitness_history) >= stagnation_limit:
            recent_history = best_fitness_history[-stagnation_limit:]
            if all(math.isclose(f, current_fitness, abs_tol=1e-15) for f in recent_history):
                print("ABBRUCH - Fitnesswert hat Stagnationslimit erreicht!\n")
                break

        fitnesses: list[float] = [fitness_function(ind) for ind in population] # Liste von Fitnesswerten jedes Individuums.
        worst_fitness_history.append(min(fitnesses))

        # Store the best performer of the current generation
        best_individual: tuple[float, float, float] = max(population, key=fitness_function) # z.B. (3.047705108690465, 4.483237638241768, -47.23145655254047)
        genes_of_best_individuals.append(best_individual)
        best_fitness: float = max(fitnesses)
        best_fitness_history.append(best_fitness)
        
        population = selection(population, fitnesses) # Elternpopulation

        next_population = []
        for i in range(0, len(population), 2): # Paarweises durchlaufen der Elternpopulation
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2 = crossover(parent1, parent2)

            next_population.append(mutation(child1, mutation_rate, lower_bound, upper_bound))
            next_population.append(mutation(child2, mutation_rate, lower_bound, upper_bound))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual # Elitismus - Das fiteste Individuum wird inverändert in die nächste Generation übernommen.
        population = next_population # Generationswechsel - Die neu erzeugte Generation wird zur Arbeitsgrundlage für den nächsten Schleifendurchlauf.

        current_fitness = best_fitness
        current_geneation += 1
            
        # return max(population, key=fitness_function)

    fitness_val: float = fitness_function(best_individual)
    x_vertex: float = get_x_vertix(best_individual)
    a, b, c = best_individual
    #result: tuple[int, float, float, float, float, float] = best_individual + (fitness_val, x_vertex)
    result: tuple[int, float, float, float, float, float] = (current_geneation, a, b, c, fitness_val, x_vertex)
    return result
    
    





# Parameters for the genetic algorithm
population_size = 100
mutation_rate = 0.2
target_fitness = 0.0
max_generations = 1000
stagnation_limit = 50
# Je größer die Koordinatengrenzen, desto gekrümmter erscheinen scheinbar flache Graphen.
lower_bound = -50 # negative Koordinatengrenzen, bis zu welcher noch betrachtet wird.
upper_bound = 50  # negative Koordinatengrenzen, bis zu welcher noch betrachtet wird.

# Run the genetic algorithm
best_solution = genetic_algorithm(population_size, mutation_rate, lower_bound, upper_bound, target_fitness, max_generations, stagnation_limit)
print("-"*30)
print(f"Best solution found:\nGeneration = {best_solution[0]},\na = {best_solution[1]}, b = {best_solution[2]}, c = {best_solution[3]},\nFitness = {best_solution[4]}")
print("-"*30)
# *_, x_vertex = best_solution


#--------------------------------------------------------------
# VISUALISIERUNG
#--------------------------------------------------------------

# Das beste Indiviuum plotten
fig, ax = plt.subplots()
ax.grid()
_, a, b, c, _, x_vertex = best_solution
min_x: float = x_vertex - 10000
max_x: float = x_vertex + 10000
x_values = np.linspace(min_x, max_x, 400) # x-Werte zwischen -5 und +5 mit 400 Punkten für eine glatte Kurve
y_values = a * (x_values ** 2) + b * x_values + c # y-Werte berechnen durch Einsetzen der x-Werte in Funkt. ax2+bx+c
ax.plot(x_values, y_values)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Quadratic Function')



# Veränderung der Gene der besten Individuen über Generationen anzeigen.
fig, ax = plt.subplots()
a_values: list[float] = [gene[0] for gene in genes_of_best_individuals]
#a_values: list[float] = [a for a, _, _ in genes_of_best_individuals]
b_values: list[float] = [gene[1] for gene in genes_of_best_individuals]
c_values: list[float] = [gene[2] for gene in genes_of_best_individuals]

reached_generation: int = best_solution[0]
#x_values = np.linspace(0, reached_generation, 400)
x_values = range(len(a_values))

ax.plot(x_values, a_values, label='a', color='blue')
ax.plot(x_values, b_values, label='b', color='green')
ax.plot(x_values, c_values, label='c', color='red')
ax.set_xlabel('Generation')
ax.set_ylabel('Parameter Values')
ax.set_title('Parameter Values Over Generations')
ax.legend()



# Veränderung der Fitness der besten Individuen über Generationen anzeigen.
fig, ax = plt.subplots()
max_fitness_values: list[float] = best_fitness_history
min_fitness_values: list[float] = worst_fitness_history
x_values = range(reached_generation)

ax.plot(x_values, max_fitness_values, label='Best Fitness', color='black')
ax.fill_between(x_values, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Over Generations')
ax.legend()
plt.show()