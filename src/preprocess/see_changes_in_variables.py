import pandas as pd


def find_duplicates(list1):
    seen_var = []
    dup_var = []
    for var in list1:
        if var not in seen_var:
            seen_var.append(var)
        else:
            dup_var.append(var)
    return dup_var


def find_var_error(list1, list2):
    """Returns things that are in list1 but not in list2
    """
    ok_cols = []
    var_error = []
    for var in list1:
        if var not in list2:
            var_error.append(var)
        else:
            ok_cols.append(var)
    return var_error


def find_ok_vars(list1, list2):
    """Returns things that are in list1 and in list2
    """
    ok_cols = []
    var_error = []
    for var in list1:
        if var not in list2:
            var_error.append(var)
        else:
            ok_cols.append(var)
    return ok_cols


if __name__ == '__main__':
    old_raw_data = pd.read_csv(f"../../data/Dataset012322.csv", dtype='str')
    old_raw_data = old_raw_data.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"Old Raw Data Shape: {old_raw_data.shape}")
    old_raw_cols = old_raw_data.columns.tolist()

    raw_data = pd.read_csv(f"../../data/Dataset052122.csv", dtype='str')
    raw_data = raw_data.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"New Raw Data Shape: {raw_data.shape}")
    raw_cols = raw_data.columns.tolist()
    # print(raw_data)

    old_preprocessed_data = pd.read_csv(
        f"../../output/preprocessed_data_without_temporal_partialfix.txt", dtype='str')
    old_preprocessed_data = old_preprocessed_data.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"Old Preprocessed Data Shape: {old_preprocessed_data.shape}")
    old_preprocessed_cols = old_preprocessed_data.columns.tolist()
    old_preprocessed_cols_no_timestamp = [
        x.rsplit('_', 1)[0] for x in old_preprocessed_cols]

    preprocessed_data = pd.read_csv(
        f"../../output/preprocessed_data_without_temporal_12to18.csv", dtype='str')
    preprocessed_data = preprocessed_data.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"Preprocessed Data Shape: {preprocessed_data.shape}")
    preprocessed_cols = preprocessed_data.columns.tolist()
    preprocessed_cols_no_timestamp = [
        x.rsplit('_', 1)[0] for x in preprocessed_cols]

    old_data_df = pd.read_csv(
        f"../../data/Variables013122new.csv",
        encoding='unicode-escape')
    old_data_df = old_data_df.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"Old Variables Data Shape: {old_data_df.shape}")
    old_data_cols = old_data_df['VariableName'].tolist()

    variables_df = pd.read_csv(
        f"../../data/Variables052122.csv",
        encoding='utf-8-sig')
    variables_df = variables_df.applymap(lambda x: x.strip() if isinstance(
        x, str) else x)
    print(f"New Variables Data Shape: {variables_df.shape}")
    data_cols = variables_df['VariableName'].tolist()
    # print(variables_df)

    # ok_cols = []
    # var_error = []
    # for var in data_cols:
    #     if var not in raw_cols:
    #         var_error.append(var)
    #     else:
    #         ok_cols.append(var)
    print(
        f"Variables listed in new encoding but not in new raw: {find_var_error(data_cols, raw_cols)}")

    # old_ok_cols = []
    # var_old_error = []
    # for var in old_data_cols:
    #     if var not in raw_cols:
    #         var_old_error.append(var)
    #     else:
    #         old_ok_cols.append(var)
    print(
        f"Variables listed in old encoding but not in new raw: {find_var_error(old_data_cols, raw_cols)}")

    # ok_cols_new_encode_old_raw = []
    # var_error_oldraw = []
    # for var in data_cols:
    #     if var not in old_raw_cols:
    #         var_error_oldraw.append(var)
    #     else:
    #         ok_cols_new_encode_old_raw.append(var)
    print(
        f"Variables listed in new encoding but not in old raw: {find_var_error(data_cols, old_raw_cols)}")

    # ok_cols_old_encode_old_raw = []
    # var_old_error_oldraw = []
    # for var in old_data_cols:
    #     if var not in old_raw_cols:
    #         var_old_error_oldraw.append(var)
    #     else:
    #         ok_cols_old_encode_old_raw.append(var)
    print(
        f"Variables listed in old encoding but not in old raw: {find_var_error(old_data_cols, old_raw_cols)}")

    # vars_new_not_old = []
    # vars_new_and_old = []
    # for var in data_cols:
    #     if var not in old_data_cols:
    #         vars_new_not_old.append(var)
    #     else:
    #         vars_new_and_old.append(var)
    print(
        f"Variables listed in new encoding but not in old: {find_var_error(data_cols, old_data_cols)}")

    # vars_old_not_new = []
    # vars_old_and_new = []
    # for var in old_data_cols:
    #     if var not in data_cols:
    #         vars_old_not_new.append(var)
    #     else:
    #         vars_old_and_new.append(var)
    print(
        f"Variables listed in old encoding but not in new: {find_var_error(old_data_cols, data_cols)}")

    print(
        f"Variables listed in new encoding more than once: {find_duplicates(data_cols)}")

    print(
        f"Variables listed in old encoding more than once: {find_duplicates(old_data_cols)}")

    # Check difference in old/new preprocessed data and old/new encoding?
    print(
        f"Variables listed in old encoding but not in old preprocessed: {find_var_error(old_data_cols, old_preprocessed_cols_no_timestamp)}")

    print(
        f"Variables listed in new encoding but not in old preprocessed: {find_var_error(data_cols, old_preprocessed_cols_no_timestamp)}")

    print(
        f"Variables listed in old encoding but not in new preprocessed: {find_var_error(old_data_cols, preprocessed_cols_no_timestamp)}")

    print(
        f"Variables listed in new encoding but not in new preprocessed: {find_var_error(data_cols, preprocessed_cols_no_timestamp)}")

    # data = pd.read_csv(f"data/Dataset012322.csv")
    # data = data[cols]
    # print(data)
    # data.to_csv('test.csv')
