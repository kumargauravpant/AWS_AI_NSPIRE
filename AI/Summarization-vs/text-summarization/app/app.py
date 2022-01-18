import torch
import base64
import json
import numpy as np
import truecase
import nltk
import tarfile
import io
import boto3
import botocore
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig
import os
# from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


def lambda_handler(event, context):

    # AWS_ACCESS_KEY_ID='AKIAZPNYRV6Q2JWEC22Q'
    # AWS_SECRET_KEY='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'
    # dynamodb = boto3.resource('dynamodb', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_KEY, region_name='us-west-2')
    S3_BUCKET = 'text-summarization-model'
    S3_KEY = 'small/'
    DYNAMO_TABLE = 'VideoAudioTransMeta'
    TEMP_PATH = '/tmp/'
    MODEL = 'text-summary-model-small.tar.gz'
    DEVICE = torch.device('cpu')

    s3_resource = boto3.resource('s3')
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table(DYNAMO_TABLE)
    userId = event['user_id']
    # userId = 'tempuserid30th'
    response = dynamo_table.get_item(Key={'user_id': userId})
    text = response['Item']['ExtractedText']
    
    # text = '''Hello. My name is Kirsten Freeman. I'm originally from Denver colorado but was raised right here in Knoxville Tennessee. Today I'll be sharing a little bit more about my background, my interest in the holidays and my plans for the future. As I mentioned before, I was raised in Knoxville. My elementary middle and high school are all within a 15 mile radius of the University of Tennessee. I've been playing sports since I was six years old, which was a crucial part of my childhood and helped me become who I am today. I'm also a very competitive person. This helped bring a sense of selflessness at a very young age because I was willing to sacrifice anything to win games and reach my desired and result this sense of selflessness would help me in the future. When I became more interested in service clubs. My interest in entering the service groups only increases. I would throughout school by my senior in high school. This interest had turned into passion. I was involved with five service clubs within my school as well as to other in the community. I received most school service principles award, a service to humanities award and other recognitions given to me by my school as well as my community for my efforts and passion in school and community service. Lastly, I would like to share my plans for the future. I'm a history major with a minor and secondary education. I plan on becoming a history teacher and participating in the teachers for America program, which combines my love of history teaching and helping others. My end goal, however, is to become a high school principal. In this position, I would love to remain heavily involved with student life. So today I've shared with you a little bit more about myself, my interest and hobbies and finally, my plans for the future. Thank you.'''
    print('text:' + text)

    # Downloading the supporting config files
    config_files = ['spiece.model', 'special_tokens_map.json', 'tokenizer_config.json', 'config.json']

    for file in config_files: 
        try:
            local_file_name = TEMP_PATH + file
            s3_resource.Bucket(S3_BUCKET).download_file(S3_KEY + file, local_file_name)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                continue
            else:
                raise 

    l = [f for f in os.listdir('/tmp/')]
    
    print(l)

    # Importing the tokenizer from temp path
    tokenizer = T5Tokenizer.from_pretrained('/tmp/')

    # Loading the config file from tmp path
    config = AutoConfig.from_pretrained(f'{TEMP_PATH}/config.json')

    # Loading the Model from S3
    s3_obj = s3_client.get_object(Bucket=S3_BUCKET, Key= S3_KEY+MODEL)
    bytestream = io.BytesIO(s3_obj['Body'].read())
    tar = tarfile.open(fileobj=bytestream, mode="r:gz")
    for member in tar.getmembers():
        if member.name.endswith(".bin"):
            f = tar.extractfile(member)
            state = torch.load(io.BytesIO(f.read()))
            model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None, state_dict=state, config=config)

    # Importing the tokenizer from temp path
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/tmp/')
    preprocess_text = text.strip().replace("\n","") 
    t5_prepared_Text = "summarize: "+preprocess_text 
    tokenized_text = tokenizer(t5_prepared_Text,return_tensors="pt").to(DEVICE)
    summary_ids = model.generate(input_ids = tokenized_text['input_ids'],
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=60,
                                    max_length=150,
                                    early_stopping=False)
    text_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print ("\n\nSummarized text: \n",text_summary)
    # text_summary = truecase.get_true_case(summary)

    # Updating summary in the dynamoDb table
    dynamo_table.update_item(
        Key={
            'user_id': userId
        },
        UpdateExpression="set text_summary=:t",
        ExpressionAttributeValues={
            ':t': text_summary
        }
    )

    print ('The text summary update is completed.')


    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": 'label',
            }
        )
    }
