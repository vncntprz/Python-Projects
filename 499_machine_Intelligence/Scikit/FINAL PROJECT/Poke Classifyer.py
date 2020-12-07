import pandas as pd
import math
import numpy as np
import operator

# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

# Import functions
import KNN as knn
import bayesian as bayes


def main():
    # prepare data
    # load pokemon dataset.
    data = pd.read_csv("Dataset/pokedex_(Update.04.20).csv")
    # Determine target
    target_col_name = 'total_points'

    # Features
    # I decided not to use 'Legendary' & 'Generation' because the variation is to low.
    # Classes like 'Bug' do not have a legendary card which results in an error: 'Divide by zero error'.
    features = ['total_points', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

    list_to_drop = ['Unnamed: 0', 'pokedex_number', 'name', 'german_name', 'japanese_name', 'generation',
                    'is_sub_legendary',
                    'is_legendary', 'is_mythical', 'species', 'type_number', 'type_1', 'type_2', 'height_m',
                    'weight_kg',
                    'abilities_number', 'ability_1', 'ability_2', 'ability_hidden', 'growth_rate', 'egg_type_number',
                    'egg_type_1', 'egg_type_2', 'percentage_male', 'egg_cycles', 'against_fairy',
                    'against_steel', 'against_dark', 'against_dragon', 'against_ghost', 'against_rock', 'against_bug',
                    'against_psychic', 'against_flying', 'against_ground', 'against_poison', 'against_fight',
                    'against_ice', 'against_ice', 'against_grass', 'against_electric', 'against_water', 'against_fire',
                    'against_normal', ]

    # this is our new DF for the algorithm
    pokePure = data.drop(list_to_drop, axis=1)

    # Split test/training dataset (80/20)
    msk = np.random.rand(len(data)) < 0.80
    train_data = data[msk]
    test_data = data[~msk]

    # Add column classify
    test_data['classify'] = test_data.index

    # Determine k = n^0.5
    k = int(math.pow(len(data.index), 0.5))

    # Get results
    test_data_knn, accuracy_knn = knn.KNN(test_data, train_data, pokePure, target_col_name, k)
    test_data_bayes, accuracy_bayes = bayes.gaussian_naive_bayes(test_data, train_data, pokePure, target_col_name)

    # Print accuracy
    print('Accuracy KNN classifier: ' + str(accuracy_knn))
    print('Accuracy Gaussian naive bayes classifier: ' + str(accuracy_bayes))


if __name__ == '__main__':
    main()
