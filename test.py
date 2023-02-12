import json
import nlp
import pandas as pd
from src.train_utils import *
from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English
from src.utils import *
from spacy.tokenizer import Tokenizer


### TEST ON VALAIDATION DATASET

def compute_rough1(predicted, reference):
    nlp_rouge = nlp.load_metric('rouge')
    scores = []
    for each_predict, each_refer in zip(predicted, reference):
        scores.append(nlp_rouge.compute(
            each_predict[0], each_refer,
            rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_agregator=True, use_stemmer=False
        )
    )
    return scores

def compute_rough_Appendix(predicted, reference):
    
    stemmer = PorterStemmer()
    rouge = Rouge()
    nlp = English()
    # Create a blank Tokenizer with just the English vocab
    tokenizer = Tokenizer(nlp.vocab)

    def compute_rouge_eli5(compare_list):
        preds = [" ".join([stemmer.stem(str(w))
                        for w in tokenizer(pred)])
                for gold, pred in compare_list]
        golds = [" ".join([stemmer.stem(str(w))
                        for w in tokenizer(gold)])
                for gold, pred in compare_list]
        scores = []        
        for pred, gold in zip(preds, golds):
            scores.append(rouge.get_scores(pred, gold))
        return scores

    compare_list = [(g, p) for p, g in zip(predicted, reference)]
    scores = compute_rouge_eli5(compare_list)
    return scores


def create_df_style(scores):
    df = pd.DataFrame({
        'rouge1': [scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']],
        'rouge2': [scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']],
        'rougeL': [scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']],
    }, index=[ 'P', 'R', 'F'])
    df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})
    return df

def get_predict_ref(dataset): 
    predicted = []
    reference = []

    # Generate answers for the full test set
    for example in dataset:
        # create support document with the dense index

        qa_s2s_tokenizer, qa_s2s_model = make_qa_s2s_model(
            model_name="facebook/bart-base",
            from_file=None,
            device="cuda:0"
        )
    
        question = example['question']
        doc = "<P> " + " <P> ".join([str(p) for p in example['ctxs']])
        # concatenate question and support document into BART input
        question_doc = "question: {} context: {}".format(question, doc)
        
        # generate an answer with beam search
        answer = qa_s2s_generate(
                                ####
                question_doc, qa_s2s_model, qa_s2s_tokenizer,
                num_answers=1,
                num_beams=8,
                min_len=96,
                max_len=256,
                max_input_length=1024,
                device="cuda:0"
        )[0]
        
        predicted += [answer]

        reference += [example['answers']]

    return

if __name__ == "__main__": 

#--------------------------------------------------------------------------------------------------

    # Val_test
    # with open(r'D:\Work\Baseline_V1\src\data\eli5\docs_cache\eli5_valid_docs_cache.json') as f:
    #     val_dataset = json.load(f)
    # Using get_predict_ref to get prediction of all sample in val_dataset:
    # prediction = get_predict_ref(val_dataset)

    # Using my saved prediction:
    with open(r'D:\Work\Baseline_V1\test\pred_ref_output_inference.json', 'r')  as f:
        pred_ref_t5 = json.load(f)

    predicted_load = pred_ref_t5[0]
    len(predicted_load)

    reference_load = pred_ref_t5[1]

    scores = compute_rough_Appendix(predicted_load, reference_load)
    print(scores)



    #SAVE TEST:

    # with open(r'D:\Work\Baseline_V1\test\10_best.txt', "w") as f:
    #     f.write()


