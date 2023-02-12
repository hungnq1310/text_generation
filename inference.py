import os
import dotenv
import psycopg2
import time
from transformers import  AutoModel, AutoTokenizer

from src.utils import *
from src.train_utils import *

dotenv.load_dotenv()

DBNAME=os.getenv("DBNAME", "wiki_db")
HOST=os.getenv("HOST", "124.158.12.207")
PORT=os.getenv("PORT", "15433")
USER=os.getenv("USER", "gradlab")
PWD=os.getenv("PASSWORD", "baldarg")
# TB_CLIENT=os.getenv("TB_CLIENT","client_tb")
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
# MSD_WIKI = bool(os.getenv("MSD_WIKI", False))

#TODO: Use this function

def query_embd(embd, limit_doc=3, ):
    embd = str(list(embd.cpu().detach().numpy().reshape(-1)))
    try:
        connection = psycopg2.connect(dbname=DBNAME,host=HOST,port=PORT,user=USER,password=PWD)
        cursor = connection.cursor()
        aemb_sql = f"""
                        SET LOCAL ivfflat.probes = 3;
                        SELECT content 
                        FROM {TB_WIKI}
                        ORDER BY embedd <#> %s LIMIT %s;
                    """
        cursor.execute(aemb_sql,(embd, limit_doc))
        connection.commit()
        rows = cursor.fetchall()

        if connection: 
            cursor.close()
            connection.close()
        
        return rows
        
    except (Exception, psycopg2.Error) as error: 
        print("Failed query record from database {}".format(error))

# def load_model_qs(
#     pretrain_name= r"vblagoje/dpr-question_encoder-single-lfqa-wiki", 
#     device = torch.device("cuda:0")
#     ):
#     qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(pretrain_name)
#     qs_model = DPRQuestionEncoder.from_pretrained(pretrain_name)
#     qs_model.to(device)
    
#     return qs_model, qs_tokenizer


def retrieve(question, qs_embedder, qs_tokenizer, device, limit_doc=10):
    # question_embd = get_ embds_qs(qs_embedder, qs_tokenizer, question, device=device)
    question_embd = get_embds_qs([question], qs_tokenizer, qs_embedder, device=device)
    documents_wiki = query_embd(question_embd, limit_doc=limit_doc)
    return [doc[-1] for doc in documents_wiki]

if __name__ == "__main__":

    qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
    qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
    _ = qar_model.eval()

    qa_s2s_tokenizer, pre_model = make_qa_s2s_model(model_name="facebook/bart-base",
                                                from_file= r"D:\Work\Baseline_V1\src\Model\bart-base_model.pth",
                                                device="cuda:0")

    while(True):
        question = input("\nUSER:")
        if question == "[EXIT]":
            break
        else:   
            doc_10 = retrieve(question, qar_model, qar_tokenizer, "cuda:0", limit_doc=10)
            doc = "<P> " + " <P> ".join([p for p in doc_10])
            question_doc = "question: {} context: {}".format(question, doc)

            # generate an answer with beam search
            # start = time.time()
            answer = qa_s2s_generate(
                    question_doc, pre_model, qa_s2s_tokenizer,
                    num_answers=1,
                    num_beams=6,
                    min_len=3,
                    max_len=100,
                    max_input_length=1024,
                    device="cuda:0")[0]       
            # end = time.time()

            print("\nBOT:", answer.replace("\n", ""))
            # print("time to generate answer: {}".format(end-start))
