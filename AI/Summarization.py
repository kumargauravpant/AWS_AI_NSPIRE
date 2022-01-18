import torch
import truecase
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

text = '''Hello. My name is Kirsten Freeman. I'm originally from Denver colorado but was raised right here in Knoxville Tennessee. Today I'll be sharing a little bit more about my background, my interest in the holidays and my plans for the future. As I mentioned before, I was raised in Knoxville. My elementary middle and high school are all within a 15 mile radius of the University of Tennessee. I've been playing sports since I was six years old, which was a crucial part of my childhood and helped me become who I am today. I'm also a very competitive person. This helped bring a sense of selflessness at a very young age because I was willing to sacrifice anything to win games and reach my desired and result this sense of selflessness would help me in the future. When I became more interested in service clubs, my interest in entering the service groups only increases I would throughout school, by my senior in high school. This interest had turned into passion. I was involved with five service clubs within my school as well as to other in the community. I received most school service principles award, a service to humanities award and other recognitions given to me by my school as well as my community for my efforts and passion in school and community service. Lastly, I would like to share my plans for the future. I'm a history major with a minor and secondary education. I plan on becoming a history teacher and participating in the teachers for America program, which combines my love of history teaching and helping others. My end goal, however, is to become a high school principal in this position, I would love to remain heavily involved with student life. So today I've shared with you a little bit more about myself, my interest and hobbies and finally, my plans for the future. Thank you.'''

model = T5ForConditionalGeneration.from_pretrained('t5-small')  

tokenizer = T5Tokenizer.from_pretrained('t5-small') 
device = torch.device('cpu')

preprocess_text = text.strip().replace("\n","") 
t5_prepared_Text = "summarize: "+preprocess_text 
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer(t5_prepared_Text,return_tensors="pt").to(device)

summary_ids = model.generate(input_ids = tokenized_text['input_ids'],
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=60,
                                    max_length=150,
                                    early_stopping=False)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)

truecase.get_true_case(output)									