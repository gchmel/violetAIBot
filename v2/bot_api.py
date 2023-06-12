import logging
import sys
from datetime import datetime

from flask import Flask
from flask_restful import Resource, Api, reqparse
from violetBot import VioletBot

app = Flask(__name__)
api = Api(app)
bot = VioletBot()

class response(Resource):
    def get(self):
        parser = reqparse.RequestParser()

        parser.add_argument('message', type=str, help='Message to send to Violet')

        args = parser.parse_args()

        return {"message" : bot.get_response(args['message'])}, 200


api.add_resource(response, '/response')

if __name__ == '__main__':
    logging.basicConfig(filename=f'example{datetime.now().date()}.log', encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    app.run()  # run our Flask app
