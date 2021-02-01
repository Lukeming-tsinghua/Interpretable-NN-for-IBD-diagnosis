import json
import os.path
import random

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import tornado.httpclient

class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, models):
        self.models = models
        self.model = None
        self.labels = ["UC", "CR", "JH"]

    def get(self):
        self.render('index.html',
                show_result=False,
                results=[],
                models=self.models)

    @tornado.gen.coroutine
    def post(self):
        texts = self.get_argument('texts').replace("\r","").split("\n")
        model_idx = int(self.get_argument('model-select'))
        self.model = self.models[model_idx]
        if len(texts) == 1 and len(texts[0]) == 0:
            self.render('index.html',
                    show_result=False,
                    results=[],
                    models=self.models)
            return
        results = []
        for text in texts:
            result = {}
            words, pred, plabel, attributions = self.model.predict(text) 
            attributions = {self.labels[i]: attributions[i] for i in range(len(self.labels))}
            result["label"] = self.labels[plabel]
            result["words"] = words
            result["attributions"] = attributions
            result["label_name"] = self.labels
            result["pred"] = pred
            results.append(result)
        self.render('index.html',
                show_result=True,
                results=results,
                models=self.models)
