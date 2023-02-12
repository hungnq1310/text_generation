import torch
from src.train_utils import *

def get_embds_qs(question, tokenizer, model, device):
    # Tokenize sentences
    q_toks = tokenizer.batch_encode_plus(question, max_length=128, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = model.embed_questions(q_ids, q_mask).cpu().type(torch.float)

    return q_reps


def qa_s2s_generate(question_doc, qa_s2s_model, qa_s2s_tokenizer, num_answers=1, num_beams=None,
                    min_len=64, max_len=256, do_sample=False,temp=1.0, top_p=None, top_k=None,
                    max_input_length=512, device="cuda:0"):
    
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer, 
                                       max_input_length, device=device)
    
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    model = qa_s2s_model.module if hasattr(qa_s2s_model, 'module') else qa_s2s_model 
    generated_ids = model.generate( input_ids=model_inputs["input_ids"],
                                           attention_mask=model_inputs["attention_mask"],
                                           min_length=min_len,max_length=max_len,
                                           do_sample=do_sample, early_stopping=True,
                                           num_beams=1 if do_sample else n_beams,
                                           temperature=temp,top_k=top_k,top_p=top_p,
                                           eos_token_id=qa_s2s_tokenizer.eos_token_id,
                                           no_repeat_ngram_size=3,
                                           num_return_sequences=num_answers,
                                           decoder_start_token_id=qa_s2s_tokenizer.bos_token_id)
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]
