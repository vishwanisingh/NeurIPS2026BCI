"""LaBraM-Base trained from random initialization on the BCI task."""

from .labram_probe import Solver as LaBraMProbeSolver


class Solver(LaBraMProbeSolver):

    name = "LaBraM-Scratch"
    pretrained = False
    train_encoder = True