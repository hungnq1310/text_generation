import torch
import torch.utils.checkpoint as checkpoint
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import math

class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(RetrievalQAEmbedder, self).__init__()
        self.sent_encoder = sent_encoder
        self.output_dim = 128
        self.project_q = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_a = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        # reproduces BERT forward pass with checkpointing
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            # prepare implicit variables
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = self.sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

            # define function for checkpointing
            def partial_encode(*inputs):
                encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask,)
                sequence_output = encoder_outputs[0]
                pooled_output = self.sent_encoder.pooler(sequence_output)
                return pooled_output

            # run embedding layer on everything at once
            embedding_output = self.sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # run encoding and pooling on one mini-batch at a time
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, q_ids, q_mask, checkpoint_batch_size=-1):
        q_reps = self.embed_sentences_checkpointed(q_ids, q_mask, checkpoint_batch_size)
        return self.project_q(q_reps)

    def embed_answers(self, a_ids, a_mask, checkpoint_batch_size=-1):
        a_reps = self.embed_sentences_checkpointed(a_ids, a_mask, checkpoint_batch_size)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=-1):
        device = q_ids.device
        q_reps = self.embed_questions(q_ids, q_mask, checkpoint_batch_size)
        a_reps = self.embed_answers(a_ids, a_mask, checkpoint_batch_size)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        loss = (loss_qa + loss_aq) / 2
        return loss

def make_qa_retriever_model(model_name="google/bert_uncased_L-8_H-512_A-8", from_file=None, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    # run bert_model on a dummy batch to get output dimension
    d_ids = torch.LongTensor(
        [[bert_model.config.bos_token_id if bert_model.config.bos_token_id is not None else 1]]
    ).to(device)
    d_mask = torch.LongTensor([[1]]).to(device)
    sent_dim = bert_model(d_ids, attention_mask=d_mask)[1].shape[-1]
    qa_embedder = RetrievalQAEmbedder(bert_model, sent_dim).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        qa_embedder.load_state_dict(param_dict["model"])
    return tokenizer, qa_embedder

def make_qa_retriever_batch(qa_list, tokenizer, max_len=64, device="cuda:0"):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=max_len, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    return (q_ids, q_mask, a_ids, a_mask)

def embed_questions_for_retrieval(q_ls, tokenizer, qa_embedder, device="cuda:0"):
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=128, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float)
    return q_reps.numpy()
