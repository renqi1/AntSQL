import jsonlines
import numpy as np
import torch
import json
import copy
from torch.utils.data import DataLoader
from utils import value_start_end

def read_train_set(path):
    """
    :return: [[question, sel, conds:[col, op, start, end], conn],...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data_list = []
        for item in jsonlines.Reader(f):
            question = item['question']
            sel = item['sql']['sel'][0]
            cond_conn_op = item['sql']['cond_conn_op']
            if item['sql'].get('conds') != None:
                conds = item['sql']['conds']
                for i, cond in enumerate(conds):
                    start, end = value_start_end(question, cond[2])
                    cond[2] = start
                    cond.append(end)
            else:
                conds=None
            data_list.append([question, sel, conds, cond_conn_op])
    return data_list

def read_test_set(path):
    """
    :return [question1, question2, ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data_list = []
        for item in jsonlines.Reader(f):
            question = item['question']
            data_list.append(question)
    return data_list

class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples1(examples, columns_encode, segment_ids, tokenizer, que_length=64, max_length=512, train=True):
    """
    :param examples: List [[ question, sel, conds=[[col, op , start, end],...], cond_conn_op],...] if train, else
                     List [question, ...]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param segment_ids: Token type ids of all colunms
    :param que_length: Maximum question length
    :param max_length: Maximum example length
    :return: [(input_ids, attention_mask, token_type_ids, label), ......]
    """
    features = []
    for example in examples:
        if train:
            question = example[0]
            label = example[1:4]
        else:
            question = example
            label = None
        input_ids = tokenizer.encode(question, add_special_tokens=True, max_length=que_length, pad_to_max_length=True)
        input_ids = input_ids + columns_encode
        token_type_ids = [0] * que_length + segment_ids
        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + ([0] * padding_length)
        input_ids = input_ids + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        features.append(InputFeatures(input_ids, attention_mask, token_type_ids, label))

    return features

def convert_examples2(examples, columns, tokenizer, que_length=64, max_length=128, train=True):
    """
    :param examples: List [[ question, sel, conds=[[col, op , start, end],...], cond_conn_op],...] if train, else
                     List[[question , cond_column_idx], ...]
    :param columns: Table columns ['基金代码', '基金名称', ...]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :return: [(input_ids, attention_mask, token_type_ids, label), ......]
    """
    features = []
    for example in examples:
        if train:
            question = example[0]
            conds = example[2]
            cond_conn_op =example[3]
            label = [conds, cond_conn_op]
            if conds is None:
                cond_column_name = None
            else:
                cond_column_name = columns[conds[0][0]]
        else:
            question = example[0]
            cond_column_name = columns[example[1]]
            label = 'test'

        inputs = tokenizer.encode_plus(question, add_special_tokens=True, max_length=que_length)
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]


        if cond_column_name is not None:
            column_name_encode = tokenizer.encode_plus(cond_column_name)
            input_ids2, token_type_ids2, attention_mask2 = column_name_encode["input_ids"], column_name_encode["token_type_ids"], column_name_encode["attention_mask"]
            input_ids = input_ids + input_ids2
            attention_mask = attention_mask + attention_mask2
            token_type_ids = token_type_ids + [1]*len(token_type_ids2)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        features.append(InputFeatures(input_ids, attention_mask, token_type_ids, label))
    return features

class BuildDataSet1(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        if feature.label is not None:
            label_sel = np.array(feature.label[0])
            label_conn = np.array(feature.label[2])
            label_conds = np.array(feature.label[1])
            if label_conds.any() == None:
                label_conds = np.array([[52,7,0,0]])
            label_conds_col = np.array(label_conds[0][0])
            label_conds_op = np.array(label_conds[0][1])
            return input_ids, attention_mask, token_type_ids, label_sel, label_conn, label_conds_col, label_conds_op
        else:
            return input_ids, attention_mask, token_type_ids

    def __len__(self):
        return len(self.features)

class BuildDataSet2(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        if feature.label == 'test':
            return input_ids, attention_mask, token_type_ids
        else:
            label_conds = np.array(feature.label[0])
            label_conds_value = self.encoder_se(label_conds)
            label_conn = np.array(feature.label[1])

            return input_ids, attention_mask, token_type_ids, label_conds_value, label_conn

    def __len__(self):
        return len(self.features)


    def encoder_se(self, conds, ques_len=63):
        """
        :param conds: [[col, op, start, end],[...]]
        :return  values: 63 * 2(start, end)
        """
        values = np.zeros((ques_len,2))
        values = np.float32(values)
        if conds.any() == None:
            return values
        else:
            start = conds[:, 2]
            end = conds[:, 3]
            for i in range(start.shape[0]):
                if start[i] == 0 and end[i] == 0:
                    return values
                values[start[i], 0] = 1
                values[end[i], 1] = 1
            return values

    def encoder_crf(self, conds, ques_len=63):
        """
        :param conds: [[col, op, start, end],[...]]
        :return  values: ques_len
        """
        values = np.zeros(ques_len)
        values = np.float32(values)
        if conds.any() == None:
            return values
        else:
            start = conds[:, 2]
            end = conds[:, 3]
            for i in range(start.shape[0]):
                if start[i] == 0 and end[i] == 0:
                    return values
                values[start[i]] = 1
                values[start[i]+1: end[i]+1] = 2
            return values
        
    def encoder_beio(self, conds, ques_len=63):
        """
        :param conds: [[col, op, start, end],[...]]
        :return  values: 63 * 4(begin, end , in ,out)
        """
        values = np.zeros((ques_len,4))
        values[:, 3] = 1
        values = np.float32(values)
        if conds.any() == None:
            return values
        else:
            start = conds[:, 2]
            end = conds[:, 3]
            for i in range(start.shape[0]):
                if start[i] == 0 and end[i] == 0:
                    return values
                values[start[i], 0] = 1
                values[end[i], 1] = 1
                values[start[i]+1:end[i], 2] = 1
            values[:, 3] = 1
            values[:, 3] = values[:, 3] - values[:, 2] - values[:, 1] - values[:, 0]
            return values

