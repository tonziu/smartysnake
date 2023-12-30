from network import Network
from game import Game
import numpy as np
import pickle


def select_mating_pool(networks, fitness, num_parents):
    parents = []
    for i in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents.append(networks[max_fitness_idx])
        fitness[max_fitness_idx] = -999999999
    return parents


def sbx_crossover(parent1, parent2, eta=1, probability=0.9):
    assert parent1.shape == parent2.shape, "Parent arrays must have the same shape."

    if np.random.rand() > probability:
        # If random number is greater than crossover probability, return parents unchanged
        return parent1.copy(), parent2.copy()

    # SBX crossover
    u = np.random.random(parent1.shape)
    beta = np.empty_like(u)

    # Apply crossover for each element in the arrays
    beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
    beta[u > 0.5] = (1 / (2 * (1 - u[u > 0.5]))) ** (1 / (eta + 1))

    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    return child1, child2


def mutation(offspring_crossover, mutation_intensity):
    # mutating the offsprings generated from crossover to maintain variation in the population
    for idx in range(offspring_crossover.shape[0]):
        for i in range(offspring_crossover.shape[1]):
            if np.random.uniform(0, 1) < mutation_intensity:
                random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=1, replace=False)
                offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
    return offspring_crossover


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def evaluate_networks(networks, game):
    scores = []
    for network in networks:
        score = game.play_ai(network)
        scores.append(score)
    return scores


def select_networks(networks, fitness):
    n = len(networks)
    networks, fitness = zip(*sorted(zip(networks, fitness), key=lambda x: x[1]))
    rank_sum = n*(n+1)/2
    probs = []
    for rank, _ in enumerate(fitness, 1):
        probs.append(float(rank) / rank_sum)
    parents = np.random.choice(networks, p=probs, size=n)
    return parents


def cross_networks(parents, offspring_count):
    children = []
    for _ in range(offspring_count):
        while True:
            i1 = np.random.randint(0, len(parents))
            i2 = np.random.randint(0, len(parents))
            if i1 != i2:
                break

        p1 = parents[i1]
        p2 = parents[i2]
        c1 = Network.copy_from(p1)
        c2 = Network.copy_from(p2)

        for w1, w2 in zip(p1.weights, p2.weights):
            wc1, wc2 = sbx_crossover(w1, w2)
            c1.weights.append(wc1)
            c2.weights.append(wc2)

        for b1, b2 in zip(p1.biases, p2.biases):
            bc1, bc2 = sbx_crossover(b1, b2)
            c1.biases.append(bc1)
            c2.biases.append(bc2)

        children.append(c1)
        children.append(c2)
    return children


def mutate_networks(networks, mutation_prob):
    for network in networks:
        for i in range(network.num_layers):
            network.weights[i] = mutation(network.weights[i], mutation_prob)
            network.biases[i] = mutation(network.biases[i], mutation_prob)
    return networks


def calculate_improvement(x, x0):
    return ((x-x0) / x0) * 100


def main(train=True):

    if not train:
        with open('best_net.pickle', 'rb') as f:
            best_network = pickle.load(f)
        game = Game.default()
        game.showcase(best_network)
        return

    size_pop = 500
    networks = [Network.default() for _ in range(size_pop)]
    for n in networks:
        n.init_layers()

    game = Game.default()

    done = False
    counter = 0
    crossover_percentage = 0.2
    mutation_prob = 0.1

    num_parents_mating = round(crossover_percentage * size_pop)
    max_fitness = 0
    best_network = None
    try:
        while not done:
            counter += 1
            fitness = evaluate_networks(networks, game)
            if np.max(fitness) > max_fitness:
                max_fitness = np.max(fitness)
                best_network = networks[np.argmax(fitness)]
            if 1:
                if counter % 50 == 0:
                    game.showcase(best_network)
            print("### -------------")
            print(f"### Gen {counter} --- Current max: {np.max(fitness):.2f} --- Overall max: {max_fitness:.2f}")
            parents = select_mating_pool(networks, fitness, num_parents_mating)
            offspring = cross_networks(parents, size_pop - num_parents_mating)
            offspring_mutation = mutate_networks(offspring, mutation_prob)
            networks[0:len(parents)] = parents
            networks[len(parents):] = offspring_mutation

    except KeyboardInterrupt:
        print("### Interrupted.")

    with open('best_net.pickle', 'wb') as f:
        pickle.dump(best_network, f)


if __name__ == "__main__":
    main(train=False)
