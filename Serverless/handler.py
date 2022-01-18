try:
    import unzip_requirements
except ImportError:
    pass
from model.model import ServerlessModel
import json

model = ServerlessModel('./model', 'text-summarization-model', 'small/text-summary-model-small.tar.gz')

def summarize_text(event, user_id):
    try:
        user_id = event['user_id']
        response = model.summarize(user_id)

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'response': response})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }