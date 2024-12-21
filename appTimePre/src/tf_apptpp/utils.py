from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pad_sequences = preprocessing.sequence.pad_sequences

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def read_data(app_train_file, app_test_file, location_train_file, location_test_file, time_train_file, time_test_file, user_info_file):

    with open(app_train_file, 'r') as in_file:
        appTrain = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    with open(app_test_file, 'r') as in_file:
        appTest = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    with open(location_train_file, 'r') as in_file:
        locationTrain = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    with open(location_test_file, 'r') as in_file:
        locationTest = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    with open(time_train_file, 'r') as in_file:
        timeTrain = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    with open(time_test_file, 'r') as in_file:
        timeTest = [[int(x.strip().split()[i]) for i in range(len(x.strip().split())) if i > 0] for x in in_file]

    assert len(timeTrain) == len(appTrain)
    assert len(locationTrain) == len(appTrain)
    assert len(appTest) == len(timeTest)

    unique_samples = set()

    for x in appTrain + appTest:
        unique_samples = unique_samples.union(x)

    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))

    appTrainIn = [x[:-1] for x in appTrain]
    appTrainOut = [x[1:] for x in appTrain]
    locationTrainIn = [x[:-1] for x in locationTrain]
    locationTrainOut = [x[1:] for x in locationTrain]
    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTrain]
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTrain]

    train_app_in_seq = pad_sequences(appTrainIn, padding='post')
    train_app_out_seq = pad_sequences(appTrainOut, padding='post')
    train_location_in_seq = pad_sequences(locationTrainIn, padding='post')
    train_location_out_seq = pad_sequences(locationTrainOut, padding='post')
    train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
    train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')

    appTestIn = [x[:-1] for x in appTest]
    appTestOut = [x[1:] for x in appTest]
    locationTestIn = [x[:-1] for x in locationTest]
    locationTestOut = [x[1:] for x in locationTest]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTest]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTest]

    test_app_in_seq = pad_sequences(appTestIn, padding='post')
    test_app_out_seq = pad_sequences(appTestOut, padding='post')
    test_location_in_seq = pad_sequences(locationTestIn, padding='post')
    test_location_out_seq = pad_sequences(locationTestOut, padding='post')
    test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
    test_time_out_seq = pad_sequences(timeTestOut, dtype=float,  padding='post')

    user_dict = {}
    with open(user_info_file, 'r') as f:
        while True:
            lines = f.readline().strip()
            if not lines:
                break
            data_arr = lines.strip().split(" ")
            user_id = int(data_arr[0])
            age = data_arr[1]
            sex = data_arr[2]
            user_dict[user_id] = (int(age), int(sex))

    with open(app_train_file, 'r') as in_file:
        train_user = [[int(x.strip().split()[0]) for i in range(len(x.strip().split())) if i == 0] for x in in_file]

    train_user_matrix = []
    for train_u in train_user:
        user_id = int(train_u[0])
        demogra = user_dict[train_u[0]]
        u_arr = [user_id, demogra[0], demogra[1]]
        train_user_matrix.append(u_arr)

    with open(app_test_file, 'r') as in_file:
        test_usre = [[int(x.strip().split()[0]) for i in range(len(x.strip().split())) if i == 0] for x in in_file]

    test_user_matrix = []
    for test_u in test_usre:
        user_id = int(test_u[0])
        demogra = user_dict[test_u[0]]
        u_arr = [user_id, demogra[0], demogra[1]]
        test_user_matrix.append(u_arr)

    unique_locations = set()
    for x in locationTrain + locationTest:
        unique_locations = unique_locations.union(x)

    return {
        'train_app_in_seq': train_app_in_seq,
        'train_app_out_seq': train_app_out_seq,

        'train_location_in_seq': train_location_in_seq,
        'train_location_out_seq': train_location_out_seq,

        'train_time_in_seq': train_time_in_seq,
        'train_time_out_seq': train_time_out_seq,

        'test_app_in_seq': test_app_in_seq,
        'test_app_out_seq': test_app_out_seq,

        'test_location_in_seq': test_location_in_seq,
        'test_location_out_seq': test_location_out_seq,

        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,

        'app_categories': len(unique_samples),
        'location_number': len(unique_locations),
        'user_number': len(user_dict),

        'train_user_info': np.array(train_user_matrix),
        'test_user_info': np.array(test_user_matrix)
    }

# delete
def calc_base_rate(data, training=True):
    suffix = 'train' if training else 'test'
    in_key = suffix + '_time_in_seq'
    out_key = suffix + '_time_out_seq'
    valid_key = suffix + '_app_in_seq'

    dts = (data[out_key] - data[in_key])[data[valid_key] > 0]
    return 1.0 / np.mean(dts)

# delete
def calc_base_app_prob(data, training=True):
    dict_key = 'train_app_in_seq' if training else 'test_app_in_seq'

    class_count = defaultdict(lambda: 0.0)
    for evts in data[dict_key]:
        for ev in evts:
            class_count[ev] += 1.0

    total_apps = 0.0
    probs = []
    for cat in range(0, data['app_categories']):
        total_apps += class_count[cat]

    for cat in range(0, data['app_categories']):
        probs.append(class_count[cat] / total_apps)

    return np.array(probs)

# delete
def data_stats(data):
    train_valid = data['train_app_in_seq'] > 0
    test_valid = data['test_app_in_seq'] > 0

    print('Num categories = ', data['app_categories'])
    print('delta-t (training) = ')
    print(pd.Series((data['train_time_out_seq'] - data['train_time_in_seq'])[train_valid]).describe())
    train_base_rate = calc_base_rate(data, training=True)
    print('base-rate = {}, log(base_rate) = {}'.format(train_base_rate, np.log(train_base_rate)))
    print('Class probs = ', calc_base_app_prob(data, training=True))

    print('delta-t (testing) = ')
    print(pd.Series((data['test_time_out_seq'] - data['test_time_in_seq'])[test_valid]).describe())
    test_base_rate = calc_base_rate(data, training=False)
    print('base-rate = {}, log(base_rate) = {}'.format(test_base_rate, np.log(test_base_rate)))
    print('Class probs = ', calc_base_app_prob(data, training=False))

    print('Training sequence lenghts = ')
    print(pd.Series(train_valid.sum(axis=1)).describe())

    print('Testing sequence lenghts = ')
    print(pd.Series(test_valid.sum(axis=1)).describe())

def MAE(time_preds, time_true, apps_out):

    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_apps_out = apps_out[:, :seq_limit]

    is_finite = np.isfinite(time_preds) & (clipped_apps_out > 0)

    return np.mean(np.abs(time_preds - clipped_time_true)[is_finite]), np.sum(is_finite)

def Recall_K(app_preds, app_true, K):

    clipped_app_true = app_true[:, :app_preds.shape[1]]
    is_valid = clipped_app_true > 0

    highest_prob_ev = app_preds.argsort(axis=-1)[:, :, ::-1][:, :, :K] + 1

    out_size_r = clipped_app_true.shape[0]
    out_size_c = clipped_app_true.shape[1]
    true_false = np.array([[clipped_app_true[r][c] in highest_prob_ev[r][c] for c in range(out_size_c)] for r in range(out_size_r)])

    return np.sum(true_false[is_valid]) / np.sum(is_valid)

def test_data_split_batch(data, batch_size, n, bptt, seed):
    test_app_in_seq = data['test_app_in_seq']
    test_location_in_seq = data['test_location_in_seq']
    test_time_in_seq = data['test_time_in_seq']
    test_user_info = data['test_user_info']

    test_app_out_seq = data['test_app_out_seq']
    test_time_out_seq = data['test_time_out_seq']
    # test_location_out_seq = testing_data['test_location_out_seq']

    idxes = list(range(len(test_app_in_seq)))
    n_batches = len(idxes) // batch_size
    for epoch in range(0, n):
        # np.random.RandomState(seed).shuffle(idxes)
        print("Starting epoch...", epoch)
        for batch_idx in range(n_batches):
            batch_idxes = idxes[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_app_test_in = test_app_in_seq[batch_idxes, :]
            batch_location_test_in = test_location_in_seq[batch_idxes, :]
            # batch_location_test_out = test_location_out_seq[batch_idxes, :]
            batch_time_test_in = test_time_in_seq[batch_idxes, :]
            batch_test_user_info = test_user_info[batch_idxes, :]  # BATCH_SIZE * 3

            batch_app_test_out = test_app_out_seq[batch_idxes, :]
            batch_time_test_out = test_time_out_seq[batch_idxes, :]

            app_ture = []
            time_true = []
            for bptt_idx in range(0, len(batch_app_test_out[0]) - bptt, bptt):
                bptt_range = range(bptt_idx, (bptt_idx + bptt))

                bptt_app_out = batch_app_test_out[:, bptt_range][:, -1]
                bptt_time_out = batch_time_test_out[:, bptt_range][:, -1]

                app_ture.append(bptt_app_out)
                time_true.append(bptt_time_out)

            batch_size_data = {
                'test_app_in_seq': batch_app_test_in,
                'test_location_in_seq': batch_location_test_in,
                'test_time_in_seq': batch_time_test_in,
                'test_user_info': batch_test_user_info,

                'test_app_out_seq_pre': np.array(app_ture).T,
                'test_time_out_seq_pre': np.array(time_true).T,
                'test_app_out_seq': batch_app_test_out,
                'test_time_out_seq': batch_time_test_out
            }

            yield batch_size_data

def train_data_split_batch(data, batch_size, n, bptt, seed):
    train_app_in_seq = data['train_app_in_seq']
    train_location_in_seq = data['train_location_in_seq']
    train_time_in_seq = data['train_time_in_seq']
    train_user_info = data['train_user_info']

    train_app_out_seq = data['train_app_out_seq']
    train_time_out_seq = data['train_time_out_seq']
    # train_location_out_seq = training_data['train_location_out_seq']

    idxes = list(range(len(train_app_in_seq)))
    n_batches = len(idxes) // batch_size

    for epoch in range(0, n):
        np.random.RandomState(seed).shuffle(idxes)

        print("Starting epoch...", epoch)

        for batch_idx in range(n_batches):

            batch_idxes = idxes[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_app_train_in = train_app_in_seq[batch_idxes, :]
            batch_location_train_in = train_location_in_seq[batch_idxes, :]
            # batch_location_train_out = train_location_out_seq[batch_idxes, :]
            batch_time_train_in = train_time_in_seq[batch_idxes, :]
            batch_train_user_info = train_user_info[batch_idxes, :]  # BATCH_SIZE * 3

            batch_app_train_out = train_app_out_seq[batch_idxes, :]
            batch_time_train_out = train_time_out_seq[batch_idxes, :]

            app_ture = []
            time_true = []
            for bptt_idx in range(0, len(batch_app_train_out[0]) - bptt, bptt):
                bptt_range = range(bptt_idx, (bptt_idx + bptt))

                bptt_app_out = batch_app_train_out[:, bptt_range][:, -1]
                bptt_time_out = batch_time_train_out[:, bptt_range][:, -1]

                app_ture.append(bptt_app_out)
                time_true.append(bptt_time_out)

            batch_size_data = {
                'train_app_in_seq': batch_app_train_in,
                'train_location_in_seq': batch_location_train_in,
                'train_time_in_seq': batch_time_train_in,
                'train_user_info': batch_train_user_info,

                'train_app_out_seq_pre': np.array(app_ture).T,
                'train_time_out_seq_pre': np.array(time_true).T,

                'train_app_out_seq': batch_app_train_out,
                'train_time_out_seq': batch_time_train_out
            }

            yield batch_size_data

if __name__ == '__main__':


    app_train_file = "E:/ATPP/code/appTimePre/data/app/data/app/apps-train.txt"
    location_train_file = ".E:/ATPP/code/appTimePre/data/app/data/app/locations-train.txt"
    time_train_file = "E:/ATPP/code/appTimePre/data/app/data/app/times-train.txt"
    app_test_file = "E:/ATPP/code/appTimePre/data/app/data/app/apps-test.txt"
    location_test_file = "E:/ATPP/code/appTimePre/data/app/data/app/app/locations-test.txt"
    time_test_file = "E:/ATPP/code/appTimePre/data/app/data/app/times-test.txt"
    user_info_file = "E:/ATPP/code/appTimePre/data/app/data/app/users.txt"

    # data = read_data(
    #     app_train_file=app_train_file,
    #     app_test_file=app_test_file,
    #     location_train_file=location_train_file,
    #     location_test_file=location_test_file,
    #     time_train_file=time_train_file,
    #     time_test_file=time_test_file,
    #     user_info_file=user_info_file
    # )
    #
    # for batch_size_data in train_data_split_batch2(data, 32, 1, 20, 1994):
    #     print(batch_size_data)

    # data_stats(data)


