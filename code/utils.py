import numpy as np

def value_start_end(question, value):
    """
    get the start and end index of the value in the question
    """
    lq = len(question)
    lv = len(value)
    for i in range(lq-lv+1):
        if question[i:lv+i] == value:
            return i, i+lv-1
    return 0, 0

def get_columns():
    columns = ['基金代码', '基金名称', '成立时间', '基金类型', '基金规模', '销售状态', '是否可销售', '风险等级', '基金公司名称', '分红方式',
               '赎回状态', '是否支持定投', '净值同步日期', '净值', '成立以来涨跌幅', '昨日涨跌幅', '近一周涨跌幅', '近一个月涨跌幅', '近三个月涨跌幅', '近六个月涨跌幅',
               '近一年涨跌幅', '基金经理', '主题/概念', '一个月夏普率', '一年夏普率', '三个月夏普率', '六个月夏普率', '成立以来夏普率', '投资市场', '板块', '行业',
               '晨星三年评级', '管理费率', '销售服务费率', '托管费率', '认购费率', '申购费率', '赎回费率', '分红年度', '权益登记日',
               '除息日', '派息日', '红利再投日', '每十份收益单位派息', '主投资产类型', '基金投资风格描述', '估值', '是否主动管理型基金', '投资', '跟踪指数',
               '是否新发', '重仓', '无']
    return columns

def encode_columns(columns, tokenizer):
    """
    get the input_ids of all columns name
    :return input_ids and segment_ids
    """
    columns_encode=[]
    segment_ids = []
    i=1
    for column in columns:
        encod = tokenizer.encode(column)
        seg = [i] * len(encod)
        columns_encode.extend(encod)
        segment_ids.extend(seg)
        if i == 1:
            i = 0
        else:
            i = 1
    return columns_encode, segment_ids

def get_cls_idx(columns, ques_length=64):
    """
    get the index of [CLS] in input_ids
    """
    gap = []
    for i in columns:
        gap.append(len(i)+2)
    cls_idx=[0]*len(gap)
    start=ques_length
    cls_idx[0] = start
    for i in range(len(gap)):
        if i>0:
            cls_idx[i] = cls_idx[i-1]+gap[i-1]
    return cls_idx

def decode_conds(conds):
    """
    :param conds: tensor(batch_size * 52 * 8)
    :return: list[[cond_col, cond_op],...]
    """
    conds = conds[:, :, 0:8].reshape(conds.shape[0], -1)
    _,ind = conds.topk(1, largest=True)
    cond_col = ind.cpu().numpy()//8
    cond_val = ind.cpu().numpy()%8
    conds = np.concatenate((cond_col,cond_val), axis=1)
    return conds

def get_values_beio(question, values):
    """
    :param values: List:63, contain the information of begin, end , inside, outside
    :return value1 and value2
    """
    question = question + " " * (63-len(question))
    count = values.count(3)
    if count > 61:
        return '', ''
    valu1 = ''
    valu2 = ''
    v1_got = v2_got = 0
    flag0 = 0
    for i, j in enumerate(values):
        if v1_got == 0:
            if j == 0 and flag0 == 0:
                valu1 += question[i]
                flag0 = 1
                continue
            if flag0 == 1:
                if j == 2:
                    valu1 += question[i]
                if j != 2:
                    if j == 0 or j == 3:
                        v1_got = 1
                    else:
                        valu1 += question[i]
                        v1_got = 1
                continue
        if v1_got == 1 and v2_got == 0:
            if j != 3:
                if j != 1:
                    valu2 += question[i]
                else:
                    valu2 += question[i]
                    v2_got = 1
        return valu1.strip(), valu2.strip()

def get_values_se(question, value1, value2, conn) :
    """
    :param value1: [start1, end1]
            value2: [start2, end2]
    :return real_value1 and real_value2
    """
    question = question + " " * (63-len(question))

    if conn == 0:
        real_value1 = question[value1[0]:value1[1] + 1]
        return real_value1.strip(), ''
    else:
        real_value1 = question[value1[0]:value1[1] + 1]
        real_value2 = question[value2[0]:value2[1] + 1]
        if value1[1] >= value2[0]:
            real_value2 = ''
        return real_value1.strip(), real_value2.strip()

def get_values_crf(question, values):
    """
    :param values: List:63, contain the information of begin, inside, outside
    :return value1 and value2
    """
    question = question + " " * (63-len(question))
    if values.count(1) == 0 or (values.count(0)==61 and values.count(2)==0):
        return '', ''
    valu1 = ''
    valu2 = ''
    v1_got = v2_got = 0
    flag1 = 0

    for i, j in enumerate(values):
        if v1_got == 0:
            if j == 1 and flag1 == 0:
                valu1 += question[i]
                flag1 = 1
                continue
            if flag1 == 1:
                if j == 2:
                    valu1 += question[i]
                else:
                    v1_got = 1

        if v1_got == 1 and v2_got == 0:
            if j != 0:
                    valu2 += question[i]

        return valu1.strip(), valu2.strip()

