import copy
import random
import time
import numpy as np

from player import Player
from utils import roulette_wheel


class Evolution:
    def __init__(self, cross_over_probability=0.5):
        self.game_mode = "Neuroevolution"
        self.cross_over_probability = cross_over_probability
        self.mutation_probability = 0.3
        self.log_file = "log.txt"
        self.selection_mode = 'roulette-wheel'
        self.parent_selection_mode = 'all'

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        players.sort()

        # Logging
        print('best and worst fitness', players[0].fitness, players[-1].fitness)
        with open(self.log_file, 'a') as log_file:
            log_file.writelines(str(players[0].fitness) + ' ' + str(players[-1].fitness) + '\n')

        new_players = []
        if self.selection_mode == 'k-best':
            new_players = players[: num_players]
        elif self.selection_mode == 'roulette-wheel':
            for player in roulette_wheel(players, 'fitness', num_players):
                new_players.append(player)
        elif self.selection_mode == 'SUS':
            sum_fitness = sum([player.fitness for player in players])

        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)
        return new_players

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            with open(self.log_file, 'w+') as log_file:
                log_file.writelines('')
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            new_players = []
            if self.parent_selection_mode == 'all':
                new_players.extend([player for player in prev_players[:len(prev_players)//10]])
                for i in range(0, len(prev_players), 2):
                    if len(new_players) >= num_players:
                        break
                    player1 = prev_players[i]
                    player2 = prev_players[i+1]
                    child1, child2 = self.cross_over(player1, player2, co_type='uniform')
                    # Mutation
                    child1.mutate(self.mutation_probability)
                    child2.mutate(self.mutation_probability)
                    new_players.append(child1)
                    new_players.append(child2)

            elif self.parent_selection_mode == 'roulette-wheel':
                pass

            return new_players

    def cross_over(self, parent1: Player, parent2: Player, co_type='uniform') -> (Player, Player):
        """
        Performs a uniform or single point cross over on two parents and makes two children.
        """
        player1, player2 = parent1.clone(), parent2.clone()

        if co_type == 'uniform':
            for i, size in enumerate(player1.nn.layer_sizes):
                if i == 0:
                    continue
                for j in range(size):
                    do_cross_over = random.random() < self.cross_over_probability
                    if do_cross_over:
                        player1.swap_perceptron(player2, i-1, j)

        elif co_type == 'single_point':
            for layer_number, layer in enumerate(player1.nn.weights):
                do_cross_over = random.random() < self.cross_over_probability
                if do_cross_over:
                    start, end = random.randint(0, layer.shape[1]-1), random.randint(0, layer.shape[1]-1)
                    perceptron = start
                    while perceptron != end:
                        player1.swap_perceptron(player2, layer_number, perceptron)
                        perceptron = (perceptron+1) % layer.shape[1]

        return player1, player2

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
