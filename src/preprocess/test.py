import pandas as pd


if __name__ == '__main__':
    raw_cols = pd.read_csv(f"data/Dataset012322_header.csv").columns.tolist()
    data_cols = pd.read_csv(
        f"data/Variables013122new.csv",
        encoding='unicode_escape'
    )['VariableName'].tolist()
    print(len(data_cols))

    cols = []
    var_error = []
    for var in data_cols:
        if var not in raw_cols:
            var_error.append(var)
        else:
            cols.append(var)

    data = pd.read_csv(f"data/Dataset012322.csv")
    data = data[cols]
    print(data)
    data.to_csv('test.csv')
