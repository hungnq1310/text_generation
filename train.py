import json
import torch
import os
from data import ELI5DatasetS2S
from src import train_utils

if __name__ == "__main__":

    PATH = path = os.getcwd()

    s2s_args = train_utils.ArgumentsS2S()

    data_train = json.load(os.path.join(PATH, "\data\eli5\origin_data\ELI5_train_10_doc.json"))
    data_val = json.load(os.path.join(PATH, "\data\eli5\origin_data\ELI5_val_10_doc.json"))

    eli5_train_docs = json.load(open(os.path.join(PATH, '\data\eli5\docs_cache\eli5_train_docs_cache.json')))
    eli5_valid_docs = json.load(open(os.path.join(PATH, '\data\eli5\docs_cache\eli5_valid_docs_cache.json')))
                                       
    s2s_train_dset = ELI5DatasetS2S(data_train, document_cache=dict([(k, d) for k, d in eli5_train_docs]))
    s2s_valid_dset = ELI5DatasetS2S(data_train, document_cache=dict([(k, d) for k, d in eli5_valid_docs]), training=False)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu:0"

    qa_s2s_tokenizer, pre_model = train_utils.make_qa_s2s_model(
        model_name="facebook/bart-large",
        from_file=None,
        device=device
    )
    qa_s2s_model = torch.nn.DataParallel(pre_model)

    train_utils.train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args)




