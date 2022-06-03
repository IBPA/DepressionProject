import numpy as np
import pandas as pd

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'output']
    test_data = pd.DataFrame(columns=cols)
    # print(rng.random(4))
    # print(rng.random(4))
    data_length = 40
    test_data['0'] = rng.integers(2, size=data_length)
    test_data['1'] = rng.random(data_length)
    test_data['2'] = rng.integers(11, size=data_length)
    test_data['3'] = rng.random(data_length)
    test_data['4'] = rng.random(data_length)
    test_data['5'] = rng.random(data_length)
    test_data['6'] = rng.random(data_length)
    test_data['7'] = rng.random(data_length)
    test_data['8'] = rng.random(data_length)
    test_data['9'] = rng.random(data_length)
    test_data['output'] = rng.choice([0, 1], size=data_length, p=[0.7, 0.3])
    test_data['10'] = test_data['output']

    # delete some data so it can be imputed
    test_data.iloc[0, 1] = np.NaN
    test_data.iloc[4, 3] = np.NaN
    test_data.iloc[0, 1] = np.NaN
    test_data.iloc[20, 9] = np.NaN
    test_data.iloc[35, 8] = np.NaN

    print(test_data)
    test_data.to_csv('./output/test_data.csv', index=False)
