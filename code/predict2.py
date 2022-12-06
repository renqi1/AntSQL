import json
import torch
from Bert import Bert2
from dataset import BuildDataSet2, convert_examples2, read_test_set
from utils import get_columns
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.autograd import Variable

columns = get_columns()
test_set_path = '../data/waic_nl2sql_testa_public.jsonl'
pretrain_vocab_path = "../pretrain_model/ernie/vocab.txt"
pretrain_config_path = "../pretrain_model/ernie/bert_config.json"
pretrain_model_path = "../pretrain_model/ernie/pytorch_model.bin"
model2_path = "../my_model/model2_ernie.pkl"
predict2_save_path = "../predict_result/predict2_ernie.json"  # predict2 according to predict1
predict1_path = "../predict_result/predict1_ernie.json"
hidden_size = 768
batch_size = 1

# test_examples = ['涨幅最大的板块南方誉隆一年持有期混合a的净值',
# '华宝消费的净值现回撤快10个',
# '产品的净值',
# '光大保德信银发商机主题混合型证券投资基金的净值推荐其他好的板块',
# '短期看好广发睿升c的净值',]

with open(predict1_path, 'r') as f:
    for i, j in enumerate(f):
        if i == 0:
            pre_all_sel_col = json.loads(j)
        elif i == 1:
            pre_all_conds_col = json.loads(j)
        elif i == 2:
            pre_all_conds_op = json.loads(j)

test_examples = read_test_set(test_set_path)
Test_examples = []
for i in range(len(pre_all_conds_col)):
    Test_examples.append([test_examples[i], pre_all_sel_col[i], pre_all_conds_col[i]])

tokenizer = BertTokenizer.from_pretrained(pretrain_vocab_path)

features = convert_examples2(Test_examples, columns, tokenizer, max_length=128, train=False)
test_dataset = BuildDataSet2(features)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('load data finish')

model2 = Bert2(hidden_size=hidden_size, config=pretrain_config_path, model=pretrain_model_path).cuda()
model2.load_state_dict(torch.load(model2_path))
model2.eval()

pre_all_conds_value1 = []
pre_all_conds_value2 = []
pre_all_conn = []
pre_all_tags = []

for i, (input_ids, attention_mask, token_type_ids) in enumerate(test_loader):
    input_ids = Variable(input_ids).cuda()
    attention_mask = Variable(attention_mask).cuda()
    token_type_ids = Variable(token_type_ids).cuda()

    out_conds_value, out_conn = model2(input_ids, attention_mask, token_type_ids)

    # pre_conds_value = torch.max(out_conds_value.data, 2)[1].cpu().numpy().tolist()

    _, pre_conds_value = torch.topk(out_conds_value, 2, dim=1, largest=True)
    pre_conds_value1 = pre_conds_value[:, 0, :].squeeze(1).cpu().numpy().tolist()
    pre_conds_value2 = pre_conds_value[:, 1, :].squeeze(1).cpu().numpy().tolist()

    # logist, out_conn = model2(input_ids, attention_mask, token_type_ids)
    pre_conn = torch.max(out_conn.data, 1)[1].cpu().numpy()

    # biesos_tags = model2.crf.decode(logist[0])
    # biesos_tags = biesos_tags.squeeze(0).cpu().numpy().tolist()

    pre_all_conds_value1.extend(pre_conds_value1)
    pre_all_conds_value2.extend(pre_conds_value2)
    pre_all_conn.extend(pre_conn)
    # pre_all_tags.extend(biesos_tags)

print('predict finish')

with open(predict2_save_path, 'w') as f:
    f.write(str(pre_all_conds_value1))
    f.write('\n')
    f.write(str(pre_all_conds_value2))
    f.write('\n')
    f.write(str(pre_all_conn))
