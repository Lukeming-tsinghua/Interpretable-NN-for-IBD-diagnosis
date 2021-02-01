import json
import os.path
import random

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import tornado.httpclient

from model import CNNPredictionModel, RobertaPredictionModel
from handler import IndexHandler

from tornado.options import define, options
define("port", default=5000, help="run on the given port", type=int)

if __name__ == '__main__':
    print("begin loading models")
    models = [CNNPredictionModel(), RobertaPredictionModel()]
    print("begin activating server")
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[
            (r'/', IndexHandler, {"models": models})
            ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
