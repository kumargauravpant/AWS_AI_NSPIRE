import torch
import boto3
import os
import tarfile
import io
import base64
import json
import re
import numpy as np
# import truecase
# import nltk
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
dynamoTable = dynamodb.Table('VideoAudioTransMeta')
#nltk.download('punkt')

class ServerlessModel:
    def __init__(self, model_path=None, s3_bucket=None, file_prefix=None):
        self.model, self.tokenizer, self.device = self.from_pretrained(model_path, s3_bucket, file_prefix)

    def from_pretrained(self, model_path: str, s3_bucket: str, file_prefix: str):
        model = self.load_model_from_s3(model_path, s3_bucket, file_prefix)
        tokenizer = self.load_tokenizer(model_path)
        device = torch.device('cpu')
        return model, tokenizer, device

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):
        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    state = torch.load(io.BytesIO(f.read()))
                    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None, state_dict=state, config=config)
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')
        
    def load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def encode(self, text):
        preprocess_text = text.strip().replace("\n","") 
        t5_prepared_Text = "summarize: "+preprocess_text 
        tokenized_text = self.tokenizer(t5_prepared_Text,return_tensors="pt").to(self.device)
        return tokenized_text

    def decode(self, token):
        summary = self.tokenizer.decode(token[0], skip_special_tokens=True)
        #return truecase.get_true_case(summary)
        return summary

    def get_text_from_dynamo(self, user_id):
        return dynamoTable.get_item(Key={'user_id': user_id})

    def update_dynamo_table(self, user_id, summary):
        dynamoTable.update_item(
            Key={
                'user_id': user_id
            },
            UpdateExpression="set text_summary=:t",
            ExpressionAttributeValues={
                ':t': summary
            }
        )
        return 'Summary Updated'
    
    def summarize(self, user_id):
        text = self.get_text_from_dynamo(user_id)
        tokenized_text = self.encode(text)
        summary_ids = self.model.generate(input_ids = tokenized_text['input_ids'],
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=60,
                                    max_length=150,
                                    early_stopping=False)
        text_summary = self.decode(summary_ids)
        upload_status = self.update_dynamo_table(user_id, text_summary)

        return upload_status



