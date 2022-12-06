import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from transformers import BertTokenizer
from Bert import Bert1
from dataset import BuildDataSet1, convert_examples1, read_test_set
from utils import encode_columns, get_cls_idx, get_columns

columns = get_columns()

# test_examples = ['何时大涨采掘跟家用电器的', '涨幅最大的板块南方誉隆一年持有期混合a的净值', '华宝消费的净值现回撤快10个',
# '产品的净值', '光大保德信银发商机主题混合型证券投资基金的净值推荐其他好的板块', '短期看好广发睿升c的净值', ]

test_set_path = "../data/waic_nl2sql_testa_public.jsonl"
pretrain_vocab_path = "../pretrain_model/ernie/vocab.txt"
pretrain_config_path = "../pretrain_model/ernie/bert_config.json"
pretrain_model_path = "../pretrain_model/ernie/pytorch_model.bin"
model1_path = "../my_model/model1_ernie.pkl"
predict1_save_path = "../predict_result/predict1_ernie.json"
hidden_size = 768
batch_size = 24

test_examples = read_test_set(test_set_path)
tokenizer = BertTokenizer.from_pretrained(pretrain_vocab_path)
columns_encode, columns_segment = encode_columns(columns, tokenizer)
features = convert_examples1(test_examples, columns_encode, columns_segment, tokenizer, que_length=64, max_length=512, train=False)
test_dataset = BuildDataSet1(features)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('load data finish')

clsidx = get_cls_idx(columns)
index = torch.LongTensor(clsidx).cuda()
model1 = Bert1(index=index, hidden_size=hidden_size, config=pretrain_config_path, model=pretrain_model_path).cuda()
model1.load_state_dict(torch.load(model1_path))
model1.eval()

pre_all_sel_col = []
pre_all_conds_col = []
pre_all_conds_op = []

for i, (input_ids, attention_mask, token_type_ids) in enumerate(test_loader):
    input_ids = Variable(input_ids).cuda()
    attention_mask = Variable(attention_mask).cuda()
    token_type_ids = Variable(token_type_ids).cuda()

    out_sel_col, out_conds_col, out_conds_op = model1(input_ids, attention_mask, token_type_ids)

    pre_sel_col = torch.max(out_sel_col.data, 1)[1].cpu().numpy()
    pre_conds_col = torch.max(out_conds_col.data, 1)[1].cpu().numpy()
    pre_conds_op = torch.max(out_conds_op.data, 1)[1].cpu().numpy()

    pre_all_sel_col.extend(pre_sel_col)
    pre_all_conds_col.extend(pre_conds_col)
    pre_all_conds_op.extend(pre_conds_op)
print('predict finish')

with open(predict1_save_path, 'w') as f:
    f.write(str(pre_all_sel_col))
    f.write('\n')
    f.write(str(pre_all_conds_col))
    f.write('\n')
    f.write(str(pre_all_conds_op))

print('write finish')
