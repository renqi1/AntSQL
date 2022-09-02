import json
from utils import get_values_se, get_columns

columns = get_columns()

with open('../predict_result/predict1_ernie.json', 'r') as f:
    for i, j in enumerate(f):
        if i == 0:
            pre_all_sel_col = json.loads(j)
        elif i == 1:
            pre_all_conds_col = json.loads(j)
        elif i == 2:
            pre_all_conds_op = json.loads(j)

with open('../predict_result/predict2_ernie.json', 'r') as f:
    for i, j in enumerate(f):
        if i == 0:
            pre_all_values1 = json.loads(j)
        if i == 1:
            pre_all_values2 = json.loads(j)
        elif i == 2:
            pre_all_conn = json.loads(j)


with open('../data/waic_nl2sql_testa_public.jsonl', 'r', encoding='UTF-8') as r:
    with open('../predict_result/waic_pred_2.jsonl', 'w', encoding='UTF-8') as w:
        for i, item in enumerate(r):
            item = json.loads(item)
            sel = pre_all_sel_col[i]
            sel_col_name = columns[sel]
            conds_col = pre_all_conds_col[i]
            conds_op = pre_all_conds_op[i]
            conds_conn = pre_all_conn[i]
            question = item["question"]
            val1 = pre_all_values1[i]
            val2 = pre_all_values2[i]
            value1, value2 = get_values_se(question, val1, val2, conds_conn)
            # value1, value2 = get_values_crf(question, pre_all_values[i])

            if conds_col == 52 or conds_op ==7:
                value1 = value2 = ''

            if value1 == '' and value2 == '':
                item['sql'] = {'sel': [sel], 'agg': [0], 'limit': 0, 'orderby': [], 'asc_desc': 0, 'cond_conn_op': 0,}
                item['keywords'] = {'sel_cols': [sel_col_name], 'values': [],}
                str = json.dumps(item, ensure_ascii=False)
                w.write(str)
                w.write('\n')

            elif value1 != '':
                if value2 != '':
                    conds = [[conds_col, conds_op, value1], [conds_col, conds_op, value2]]
                    values = [value1, value2]
                else:
                    conds = [[conds_col, conds_op, value1]]
                    values = [value1]
                    conds_conn = 0

                item['sql'] = {'sel': [sel], 'agg': [0], 'limit': 0, 'orderby': [], 'asc_desc': 0, 'cond_conn_op': conds_conn, 'conds': conds, }
                item['keywords'] = {'sel_cols': [sel_col_name], 'values': values,}
                str = json.dumps(item, ensure_ascii=False)
                w.write(str)
                w.write('\n')

