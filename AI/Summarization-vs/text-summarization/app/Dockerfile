FROM public.ecr.aws/lambda/python:3.8

COPY app.py requirements.txt ./
COPY model /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t .

RUN python3.8 -m nltk.downloader punkt

CMD ["app.lambda_handler"]
