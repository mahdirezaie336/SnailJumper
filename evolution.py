import copy
import random
import time

from player import Player


class Evolution:
    def __init__(self, cross_over_probability=0.5):
        self.game_mode = "Neuroevolution"
        self.cross_over_probability = cross_over_probability

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        players.sort()
        print('best and worst fitness', players[0].fitness, players[1].fitness)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)
        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            new_players = []
            for i in range(0, len(prev_players), 2):
                player1 = prev_players[i]
                player2 = prev_players[i+1]
                child1, child2 = self.cross_over(player1, player2)
                new_players.append(child1)
                new_players.append(child2)

                if len(new_players) >= num_players:
                    break

            return new_players

    def cross_over(self, parent1: Player, parent2: Player):
        """
        Performs a uniform cross over on two parents and makes two children.
        """
        player1, player2 = parent1.clone(), parent2.clone()
        for i, size in enumerate(player1.nn.layer_sizes):
            if i == 0:
                continue
            for j in range(size):
                do_cross_over = random.random() < self.cross_over_probability
                if do_cross_over:
                    player1.swap_perceptron(player2, i-1, j)
        return player1, player2

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
