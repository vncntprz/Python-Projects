'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 3
'''

import random

class Die:
    def __init__(self, sides):
        self.__sides = sides
        self.__roll = 0

    def roll(self):
        self.__roll = (random.randint(1, self.__sides))
        return self.__roll

d = Die(10)
print(d.roll())
