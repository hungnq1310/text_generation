import torch
from torch.utils.data import  Dataset

class ELI5DatasetS2S(Dataset):
    def __init__(
        self, examples_array, make_doc_fun=None, document_cache=None, training=True
    ):
        self.training = training
        self.data = examples_array
        self.make_doc_function = make_doc_fun
            
        self.document_cache = {} if document_cache is None else document_cache
        
        assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        # Fix: just using only one answer for training 
        # Not require self.training
        if self.training:
            self.qa_id_list = [
                (i, j)
                for i, qa in enumerate(self.data)
                for j, a in enumerate(qa["answers"])
            ]
        else:
            self.qa_id_list = [(i, 0) for i in range(len(self.data))]

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example["question"]
        # using only the first answer.
        answer = example["answers"][j]
        q_id = example["question_id"]
#         if self.make_doc_function is not None:
#             self.document_cache[q_id] = self.document_cache.get(q_id, self.make_doc_function(example["title"]))
        document = self.document_cache[q_id] 
        # return from dict document_cache { question_id : ctxs_dense}
        # Done
        in_st = "question: {} context: {}".format(
            question.lower().strip(), document.lower().strip(),
        )
        out_st = answer
        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(idx)

def embed_passages_for_retrieval(passages, tokenizer, qa_embedder, max_length=128, device="cuda:0"):
    a_toks = tokenizer.batch_encode_plus(passages, max_length=max_length, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        a_reps = qa_embedder.embed_answers(a_ids, a_mask).cpu().type(torch.float)
    return a_reps.numpy()

def embed_questions_for_retrieval(q_ls, tokenizer, qa_embedder, device="cuda:0"):
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=128, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float)

    print(q_ids)
    return q_reps.numpy()

    