import numpy as np
import pandas as pd

if __name__ == '__main__':
    action_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    stats = np.zeros((len(action_list), 4))
    for i, action in enumerate(action_list):
        filename = 'A{}_constrained_diffmulti.csv'.format(str(action))
        df = pd.read_csv(filename, header=None, names=['seed', 'budget', 'init', 'random', 'random violation', 'opt', 'opt violation', 'forward', 'backward'])
        stats[i,0] = np.mean(df['forward'])  / 100
        stats[i,1] = np.mean(df['backward']) / 100
        stats[i,2] = np.std(df['forward'])  / 100
        stats[i,3] = np.std(df['backward']) / 100
        print('{}, {}, {}, {}'.format(str(stats[i,0]), str(stats[i,1]), str(stats[i,2]), str(stats[i,3])))
