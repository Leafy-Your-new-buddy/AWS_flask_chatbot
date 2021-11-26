import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd

from kobert_tokenizer import KoBERTTokenizer

# from transformers import AdamW
# from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

# from kobert_transformers import get_kobert_model
# from kobert_transformers import get_tokenizer

from kobert_transformers import get_tokenizer
from kobert.pytorch_kobert import get_kobert_model
 
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)

import re
def cleanText(input):
  clean = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','',input)
  return clean

import json
with open('answer.json',encoding="UTF8") as json_file:
    answer_data = json.load(json_file)

chatbot_data = pd.read_excel('data.xlsx', engine = 'openpyxl')

tok=tokenizer.tokenize
transform = nlp.data.BERTSentenceTransform(tokenizer=tok,
                                           vocab=vocab,
                                           max_seq_length=140,
                                           pad=True,
                                           pair=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=32,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


device = torch.device("cpu")

from hanspell import spell_checker

def spellCheck(input):
    checked = spell_checker.check(input)
    return checked.as_dict()['checked']

import random 
def getAnswer(index, num):

  for tg in answer_data["intents"]:
    if tg['tag']==str(index):
      if num == 0:
        answers = tg['answer']
      elif num==1:
        answers = tg['answeryes']
      elif num==2:
        answers = tg['answerno']

  ans=answers[0]
  return ans



import pickle

import builtins
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

model.load_state_dict(torch.load('final_model.pkl',map_location=torch.device('cpu')))

#load_model=builtins.open('final_model.pkl', 'r')

# loaded_model=BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
# loaded_model.load_state_dict(torch.load('final_model.pkl',map_location=torch.device('cpu')))


def predict(predict_sentence,beforeidx):

    predict_sentence = spellCheck(predict_sentence)
    predict_sentence = cleanText(predict_sentence)

    if predict_sentence=='응' and (beforeidx==3 or beforeidx==15 or beforeidx==17 or beforeidx==18 or beforeidx==22 or beforeidx==25 or beforeidx==28 or beforeidx==31):
        answer = getAnswer(beforeidx,1)
        return answer,-1
    elif predict_sentence=='아니' and (beforeidx==3 or beforeidx==15 or beforeidx==17 or beforeidx==18 or beforeidx==22 or beforeidx==25 or beforeidx==28 or beforeidx==31):
        answer = getAnswer(beforeidx,2)
        return answer,-1
    else:
    
        data = [predict_sentence, '0']

        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok,vocab, max_len, True, False)
    
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=2) 
    
        model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)
      
            out = model(token_ids, valid_length, segment_ids)
            tmp = out.detach().cpu().numpy()
            maxProb = np.max(tmp[0])
            test_eval=[]
            idx = -1
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                idx = np.argmax(logits)

                if idx == 0:
                    test_eval.append("분갈이_방법")
                elif idx == 1:
                    test_eval.append("분갈이_시기")
                    
                elif idx == 2:
                    test_eval.append("분갈이_이후")
                    
                elif idx == 3:
                    test_eval.append("분갈이_화분")
                    beforeidx=idx
                elif idx == 4:
                    test_eval.append("분갈이_흙")
                elif idx == 5:
                    test_eval.append("잎의질병_화상")
                elif idx == 6:
                    test_eval.append( "재배환경_온도" )
                elif idx == 7:
                    test_eval.append( "재배환경_인조조명"  )
                elif idx == 8:
                    test_eval.append( "재배환경_장소")
                elif idx == 9:
                    test_eval.append("재배환경_햇빛" )
                elif idx == 10:
                    test_eval.append("튜토리얼_방법" )
                elif idx == 11:
                    test_eval.append("물주기" )
                elif idx == 12:
                    test_eval.append("물주기_저면관수" )
                elif idx == 13:
                    test_eval.append("물주기_계절" )
                    beforeidx=idx
                elif idx == 14:
                    test_eval.append( "번식_잎꽂이방법" )
                elif idx == 15:
                    test_eval.append("번식_자구" )
                    beforeidx=idx
                elif idx == 16:
                    test_eval.append(  "번식_적심")
                elif idx == 17:
                    test_eval.append("뿌리줄기_무름병" )
                    beforeidx=idx
                elif idx == 18:
                    test_eval.append("뿌리줄기_썩음병" )
                    beforeidx=idx
                elif idx == 19:
                    test_eval.append( "뿌리줄기_기타"  )
                elif idx == 20:
                    test_eval.append( "웃자람_증상"  )
                elif idx == 21:
                    test_eval.append("웃자람_해결"  )
                elif idx == 22:
                    test_eval.append( "잎의질병_곰팡이" )
                    beforeidx=idx
                elif idx == 23:
                    test_eval.append( "잎의질병_과습" )
                elif idx == 24:
                    test_eval.append( "잎의질병_과습여부")
                elif idx == 25:
                    test_eval.append( "잎의질병_마름"   )
                    beforeidx=idx
                elif idx == 26:
                    test_eval.append( "잎의질병_원인알수없음" )
                elif idx == 27:
                    test_eval.append(  "잎의질병_탄저병" )
                elif idx == 28:
                    test_eval.append( "해충_기타"   )
                    beforeidx=idx
                elif idx == 29:
                    test_eval.append(  "해충_깍지" )
                    beforeidx=idx                    
                elif idx == 30:
                    test_eval.append("해충_응애" )
                elif idx == 31:
                    test_eval.append("해충_해충약" )
                    beforeidx=idx            
             
            answer = getAnswer(idx,0)

        return answer,idx

 

from flask import Flask, url_for, request, jsonify
from flask import make_response
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def chatBot():
       global beforeidx 
       global s
       parameter_dict = request.args.to_dict()
       if len(parameter_dict) == 0:
           s="리피에게 궁금한걸 물어봐주세요~"
        
       else:
           question=request.args.get('msg')
           if question == '응':              
               s,tmp=predict(question,beforeidx)                
           elif question =='아니':               
               s,tmp=predict(question,beforeidx)                
           else:
               s,beforeidx =predict(question,-1)
       result = json.dumps({"answer": s}, ensure_ascii=False)      
       res = make_response(result)     
       return res, 200,{'content-type': 'application/json'}

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)
 

