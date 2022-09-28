import math, numpy as np, pandas as pd
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

def gen_pos_dists(data, pos_split_config, y_col):

    main_data = data
    pos_data = data[data[y_col] == 1]

    if isinstance(pos_split_config, dict):
        main_data = data[data[y_col] == 0]
        # pos_data_base = data[data[y_col] == 1]

        # pos_data_base_size = pos_data_base.shape[0]
        #
        # pos_split_config = {'train': pos_dist[0],
        #                     'val': pos_dist[1],
        #                     'test': pos_dist[2]}
        #
        # pos_data = {'train': None, 'val': None, 'test': None}
        #
        # if pos_split_config['train']:
        #     pos_data['train'], pos_data_vt = train_test_split(pos_data_base, train_size=pos_split_config['train'], random_state=random_state, shuffle=pre_shuffle)
        #
        #     pos_split_config
        #
        # else:
        #     pos_data_vt = pos_data_base
        #
        # if pos_split_config['val'] and pos_split_config['test']:
        #     pos_data['val'], pos_data['test'] = train_test_split(pos_data_vt, train_size=pos_split_config['val'], random_state=random_state, shuffle=pre_shuffle)
        #
        # elif pos_split_config['val'] and not pos_split_config['test']:
        #     pos_data['val'] = pos_data_vt
        #
        # else:
        #     pos_data['test'] = pos_data_vt

    return main_data, pos_data

def norm_scale_data(datasets):
    for k, v in datasets.items():

        norm_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler((0, 1))

        if v:
            if 'Amount' in v.x.columns:
                v.x['Amount'] = norm_scaler.fit_transform(v.x[['Amount']])
            datasets[k].x = min_max_scaler.fit_transform(v.x)

    return datasets

def process_split_config(split_config):

    if split_config:

        if sum(split_config) > 1:
            raise ValueError(f"Ivalid Split Config")

        split_config = {'train': split_config[0],
                        'val': split_config[1],
                        'test': split_config[2]}

        for k, v in split_config.items():
            if k == 'train':
                train_size = 1 - v
            else:
                split_config[k] = round(v / train_size,5)

    return split_config

def split_data(data, split_config, pre_shuffle, random_state):

    if split_config == 0:
        return None

    pos_data = {'train': None,
                'val': None,
                'test': None}

    split_data = {'train': None,
                  'val': None,
                  'test': None}

    if isinstance(data,list):
        main_data = data[0]
        if data[1]:
            pos_data = data[1]
    else:
        main_data = data

    if split_config['val'] or split_config['test']:

        if split_config['val'] and split_config['test']:

            split_data['train'], vt_data = train_test_split(main_data,
                                                            train_size=split_config['train'],
                                                            random_state=random_state,
                                                            shuffle=pre_shuffle)

            split_data['val'], split_data['test'] = train_test_split(vt_data,
                                                                     test_size=split_config['test'],
                                                                     random_state=random_state,
                                                                     shuffle=pre_shuffle)
        else:

            if split_config['val']:
                if split_config['train']:
                    split_data['train'], split_data['val'] = train_test_split(main_data,
                                                                              train_size=split_config['train'],
                                                                              random_state=random_state,
                                                                              shuffle=pre_shuffle)
                else:
                    split_data['val'] = main_data

            if split_config['test']:
                if split_config['train']:

                    split_data['train'], split_data['test'] = train_test_split(main_data,
                                                                                   train_size=split_config['train'],
                                                                                   random_state=random_state,
                                                                                   shuffle=pre_shuffle)
                else:
                    split_data['test'] = main_data
    else:

        if pre_shuffle:
            split_data['train'] = main_data.sample(frac=1)
        else:
            split_data['train'] = main_data


    for k, v in split_data.items():
        if v is not None and pos_data[k] is not None:
            split_data[k] = pd.concat([split_data[k],pos_data[k]], axis=0)

    return split_data

def preprocess_data(data,
                    cols,
                    split_config,
                    pos_split_config=None,
                    pre_shuffle=False,
                    to_numpy=False,
                    dtype=np.float32,
                    random_state=None,
                    debug_on=False):

    train_no_pos = False

    x_cols, y_col = cols[0], cols[1][0]

    split_config = process_split_config(split_config)

    if pos_split_config != 0:
        if pos_split_config:
            pos_split_config = process_split_config(pos_split_config)
        else:
            pos_split_config = split_config


    main_data = data[data[y_col] == 0]
    pos_data = data[data[y_col] == 1]
    n_pos = pos_data.shape[0]

    pos_data = split_data(pos_data,pos_split_config,pre_shuffle,random_state)

    datasets = split_data([main_data,pos_data], split_config, pre_shuffle, random_state)


    if debug_on:
        n_train_pos = datasets['train'][datasets['train'][y_col] == 1].shape[0]
        n_val_pos = datasets['val'][datasets['val'][y_col] == 1].shape[0] if datasets['val'] is not None else 0
        n_test_pos = datasets['test'][datasets['test'][y_col] == 1].shape[0] if datasets['test'] is not None else 0
        print(f"\nTrain Pos: {n_train_pos} | {n_train_pos/n_pos:.2f} || Val Pos: {n_val_pos} | {n_val_pos/n_pos:.2f} || Test Pos: {n_test_pos} | {n_test_pos/n_pos:.2f}")

    # split x, y
    for k, v in datasets.items():

        if v is not None:

            x = v[x_cols].astype(dtype)
            y = v[y_col].astype(dtype)

            if to_numpy:
                x = x.to_numpy()
                y = y.to_numpy()

            datasets[k] = ModelData(x, y)

    if debug_on:
        counts_check(datasets,y_col)

    return norm_scale_data(datasets)


