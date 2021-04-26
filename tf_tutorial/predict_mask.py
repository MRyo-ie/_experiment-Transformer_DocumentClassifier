# coding: utf-8
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, BertConfig, AutoModelForMaskedLM, AutoModel
from Experiment.ModelABC.tokenizer import Juman_Tokenizer



def get_model(model_name):
    if model_name == "T5":
        return T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
    elif model_name == "BERT":
        # return AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    elif model_name == "BERT_kyoto":
        config = BertConfig.from_json_file(
                    'model/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json')
        return AutoModel.from_pretrained(
                    'model/Japanese_L-12_H-768_A-12_E-30_BPE/pytorch_model.bin',
                    config=config)
    raise Exception(f"[Error] モデル：{model_name} は未調整です。")



def get_tokenizer(model_name):
    if model_name == "T5":
        return T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
    elif model_name == "BERT":
        # return AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    elif model_name == "BERT_kyoto":
        return AutoTokenizer.from_pretrained(
                    "../model/Japanese_L-12_H-768_A-12_E-30_BPE",
                    do_lower_case=False, do_basic_tokenize=False)
    raise Exception(f"[Error] モデル：{model_name} は未調整です。")



if __name__ == "__main__":
    # トークナイザーとモデルの準備
    model_all_names = ["T5", "BERT_kyoto", "BERT"]
    model_name = model_all_names[2]

    text = '吾輩は猫である。名前はまだない。'
    mask_idx = 2
    if len(sys.argv) > 1:
        text = sys.argv[1]
    if len(sys.argv) == 3:
        mask_idx = sys.argv[2]

    tokenizer = get_tokenizer(model_name)
    mask_token = tokenizer.mask_token
    if tokenizer.mask_token is None:
        mask_token = '[MASK]'
    print(text, mask_idx, mask_token)

    # tokenized_text = tokenizer.tokenize(text)
    #=>  ['テレビ', 'で', 'サッカー', 'の', '試合', 'を', '見る', '。']

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    # tokenized_text[mask_idx] = '[MASK]'
    # ['テレビ', 'で', '[MASK]', 'の', '試合', 'を', '見る', '。']
    
    # テキストをテンソルに変換
    input_ids = tokenizer.encode(text, return_tensors='pt')
    print(input_ids)
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]


    # 推論
    model = get_model(model_name)
    
    result = model(input_ids)
    pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
    for pred_id in pred_ids:
        output_ids = input_ids.tolist()[0]
        output_ids[masked_index] = pred_id
        print(tokenizer.decode(output_ids))


