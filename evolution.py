import copy
import random
import time
import numpy as np

from player import Player
from utils import roulette_wheel, sus, choose_pairs, q_tournament


class Evolution:
    def __init__(self, cross_over_probability=0.5):
        self.game_mode = "Neuroevolution"
        self.cross_over_probability = cross_over_probability
        self.mutation_probability = 0.3
        self.log_file = "log.txt"
        self.selection_mode = 'SUS'
        self.parent_selection_mode = 'SUS'
        self.cross_over_type = 'uniform'

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        players.sort()

        # Logging
        mean = sum([player.fitness for player in players]) / len(players)
        print('best and worst fitness, mean', players[0].fitness, players[-1].fitness, mean)
        with open(self.log_file, 'a') as log_file:
            log_file.writelines(str(players[0].fitness) + ' ' + str(players[-1].fitness) + ' ' + str(mean) + '\n')

        players_clone = [player.clone() for player in players]
        new_players = []
        if self.selection_mode == 'k-best':
            new_players = players_clone[: num_players]
        elif self.selection_mode == 'roulette-wheel':
            for player in roulette_wheel(players_clone, 'fitness', num_players):
                new_players.append(player)
        elif self.selection_mode == 'SUS':
            for player in sus(players_clone, 'fitness', num_players):
                new_players.append(player)

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
            chosen = []
            if self.parent_selection_mode == 'all':
                k = 10
                new_players.extend([player for player in prev_players[:k]])
                for player1, player2 in choose_pairs(prev_players, num_players - k):
                    child1, child2 = self.cross_over(player1, player2, co_type=self.cross_over_type)
                    child1.mutate(self.mutation_probability)
                    child2.mutate(self.mutation_probability)
                    new_players.append(child1)
                    new_players.append(child2)
                return new_players

            elif self.parent_selection_mode == 'roulette-wheel':
                chosen = [player for player in roulette_wheel(prev_players, 'fitness', num_players)]

            elif self.parent_selection_mode == 'SUS':
                chosen = [player for player in sus(prev_players, 'fitness', num_players)]

            elif self.parent_selection_mode == 'q-tournament':
                chosen = [player for player in q_tournament(prev_players, 'fitness', num_players)]

            # Cross over and mutation
            for player1, player2 in choose_pairs(chosen, num_players):
                child1, child2 = self.cross_over(player1, player2, co_type=self.cross_over_type)
                child1.mutate(self.mutation_probability)
                child2.mutate(self.mutation_probability)
                new_players.append(child1)
                new_players.append(child2)

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

        elif co_type == 'single-point':
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
