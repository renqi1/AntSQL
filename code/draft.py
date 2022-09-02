def encode_sentence(sentence, vocab_dic, is_question=False, ques_length=64):
    S=[vocab_dic['[CLS]']]
    for word in sentence:
        if word in vocab_dic:
            S.append(vocab_dic[word])
        else:
            S.append(vocab_dic['[UNK]'])
    if is_question:
        S = S + ([0] * (ques_length-1-len(S)))
    S.append(vocab_dic['[SEP]'])
    return S

def my_encode(question, columns, vocab_dic, ques_length, max_length):
    input_ids = encode_sentence(question, vocab_dic, is_question=True, ques_length=ques_length)
    for c in columns:
        input_ids.extend(encode_sentence(c,vocab_dic))

    padding_length = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    sep = vocab_dic['[SEP]']
    token_type_ids=[]
    fill = 0
    for i in input_ids:
        if i != sep:
            token_type_ids.append(fill)
        else:
            token_type_ids.append(fill)
            fill += 1
    attention_mask = attention_mask + ([0] * padding_length)

    return input_ids,  attention_mask, token_type_ids

def get_vocab_dict(vocab_path):
    with open(vocab_path, 'r', encoding='UTF-8') as f:
        vocab_dic = {}
        data = f.readlines()
        for i, word in enumerate(data):
            vocab_dic.update({word.strip():i})
    return vocab_dic

def encode_sentence1(sentence, vocab_dic, is_question=False, ques_length=64, col_name_length=10):
    S=[vocab_dic['[CLS]']]
    for word in sentence:
        if word in vocab_dic:
            S.append(vocab_dic[word])
        else:
            S.append(vocab_dic['[UNK]'])
    if is_question:
        S = S + ([0] * (ques_length-1-len(S)))
    else:
        S = S + ([0] * (col_name_length - 1 - len(S)))
    S.append(vocab_dic['[SEP]'])
    return S

def convert_examples2(examples, columns, tokenizer, que_length=64, max_length=128, train=True):
    """
    :param examples: List [[ question, sel, conds=[[col, op , start, end],...], cond_conn_op],...] if train, else
                     List[[question ,sel_column_idx, cond_column_idx], ...]
    :param columns: Table columns ['基金代码', '基金名称', ...]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :return: [(input_ids, attention_mask, token_type_ids, label), ......]
    """
    features = []
    for example in examples:
        if train:
            question = example[0]
            conds = example[2]
            sel = example[1]
            cond_conn_op =example[3]
            sel_column_name = columns[sel]
            label = [conds, cond_conn_op]
            if conds is None:
                cond_column_name = None
            else:
                cond_column_name = columns[conds[0][0]]
        else:
            question = example[0]
            sel_column_name = columns[example[1]]
            cond_column_name = columns[example[2]]
            label = 'test'

        inputs = tokenizer.encode_plus(question, add_special_tokens=True, max_length=que_length, pad_to_max_length=True)
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]

        sel_name_encode = tokenizer.encode_plus(sel_column_name)
        input_ids2, token_type_ids2, attention_mask2 = sel_name_encode["input_ids"], sel_name_encode["token_type_ids"], sel_name_encode["attention_mask"]
        input_ids = input_ids + input_ids2
        attention_mask = attention_mask + attention_mask2
        token_type_ids = token_type_ids + [1]*len(token_type_ids2)

        if cond_column_name is not None:
            column_name_encode = tokenizer.encode_plus(cond_column_name)
            input_ids3, token_type_ids3, attention_mask3 = column_name_encode["input_ids"], column_name_encode["token_type_ids"], column_name_encode["attention_mask"]
            input_ids = input_ids + input_ids3
            attention_mask = attention_mask + attention_mask3
            token_type_ids = token_type_ids + [0]*len(token_type_ids3)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        features.append(InputFeatures(input_ids, attention_mask, token_type_ids, label))
    return features