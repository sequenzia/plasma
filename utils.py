import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

@dataclass
class ModelData:
    x: Union[ndarray, DataFrame, Tensor]
    y: Union[ndarray, DataFrame, Tensor] = None

def set_options(fp=2,
                max_rows=500,
                max_cols=25,
                edgeitems=30,
                threshold=10000,
                all_cols=False,
                linewidth=900):

    float_p = '%.' + str(fp) + 'f'

    ff = lambda x: float_p % x

    pd.options.display.float_format = ff
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)

    if all_cols:
        pd.set_option('display.expand_frame_repr', False)

    np.set_printoptions(formatter={'float': ff},
                        edgeitems=edgeitems,
                        linewidth=linewidth,
                        threshold=threshold)

def counts_check(data, y_col):
    train_n_records = data['train'].x.shape[0]
    val_n_records = data['val'].x.shape[0] if data['val'] else 0
    test_n_records = data['test'].x.shape[0] if data['test'] else 0

    n_records = train_n_records + val_n_records + test_n_records

    print(f"\nTotal Records: {n_records} | Train: {(train_n_records / n_records):.2f} | Val: {(val_n_records / n_records):.2f} | Test: {(test_n_records / n_records):.2f}")

def split_data(data,
               val_test_config,
               cols,
               pre_shuffle=False,
               to_numpy=False,
               dtype=np.float32,
               pos_dist=False,
               debug_on=False):

    train_no_pos = False
    seed = 0

    x_cols, y_col = cols[0], cols[1][0]

    train_size = 1 - sum(val_test_config)

    split_config = {'train': train_size,
                    'val': val_test_config[0] / (1 - train_size) if val_test_config[0] else 0,
                    'test': val_test_config[1] / (1 - train_size) if val_test_config[1] else 0}

    # break out pos class
    pos_data = None
    main_data = data

    if isinstance(pos_dist, list):

        main_data = data[data[y_col] == 0]
        pos_data_base = data[data[y_col] == 1]
        pos_data_base_size = pos_data_base.shape[0]

        pos_split_config = {'train': pos_dist[0],
                            'val': pos_dist[1],
                            'test': pos_dist[2]}

        pos_data = {'train': None, 'val': None, 'test': None}

        if pos_split_config['train']:
            pos_data['train'], pos_data_vt = train_test_split(pos_data_base, train_size=pos_split_config['train'], random_state=seed, shuffle=pre_shuffle)

            pos_split_config

        else:
            pos_data_vt = pos_data_base

        if pos_split_config['val'] and pos_split_config['test']:
            pos_data['val'], pos_data['test'] = train_test_split(pos_data_vt, train_size=pos_split_config['val'], random_state=seed, shuffle=pre_shuffle)

        elif pos_split_config['val'] and not pos_split_config['test']:
            pos_data['val'] = pos_data_vt

        else:
            pos_data['test'] = pos_data_vt

    split_data = {'train': None,
                  'val': None,
                  'test': None,
                  'pos_data': pos_data}

    # split into train, val, test
    if split_config['val'] or split_config['test']:

        if split_config['val'] and split_config['test']:

            split_data['train'], vt_data = train_test_split(main_data, train_size=split_config['train'],
                                                            random_state=seed, shuffle=pre_shuffle)

            if train_no_pos:
                vt_data = pd.concat([vt_data, pos_data])

            split_data['val'], split_data['test'] = train_test_split(vt_data, test_size=split_config['test'],
                                                                     random_state=seed, shuffle=pre_shuffle)

        else:

            if split_config['val']:

                split_data['train'], split_data['val'] = train_test_split(main_data, train_size=split_config['train'],
                                                                          random_state=seed, shuffle=pre_shuffle)

                if train_no_pos:
                    split_data['val'] = pd.concat([split_data['val'], pos_data]).sample(frac=1)

            if split_config['test']:

                split_data['train'], split_data['test'] = train_test_split(main_data, train_size=split_config['train'],
                                                                           random_state=seed, shuffle=pre_shuffle)

                if train_no_pos:
                    split_data['test'] = pd.concat([split_data['test'], pos_data]).sample(frac=1)

    else:
        if pre_shuffle:
            split_data['train'] = main_data.sample(frac=1)
        else:
            split_data['train'] = main_data

    if debug_on:
        n_train_pos = split_data['train'][split_data['train']['Class'] == 1].shape[0]
        n_val_pos = split_data['val'][split_data['val']['Class'] == 1].shape[0] if split_data['val'] is not None else 0
        n_test_pos = split_data['test'][split_data['test']['Class'] == 1].shape[0] if split_data[
                                                                               'test'] is not None else 0
        print(f"\nTrain Pos: {n_train_pos} | Val Pos: {n_val_pos} | Test Pos: {n_test_pos}")

    # split x, y
    for k, v in split_data.items():

        if v is not None:

            x = v[x_cols].astype(dtype)
            y = v[y_col].astype(dtype)

            if to_numpy:
                x = x.to_numpy()
                y = y.to_numpy()

            split_data[k] = ModelData(x, y)

    if debug_on:
        counts_check(split_data,y_col)

    return split_data

def preprocess_data(datasets):
    for k, v in datasets.items():

        norm_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler((0, 1))

        if v:
            v.x['Amount'] = norm_scaler.fit_transform(v.x[['Amount']])
            datasets[k].x = min_max_scaler.fit_transform(v.x)

    return datasets


