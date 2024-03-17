from ArchiClassifierModel import ArchiClassifierModel
import logging
import fire


class CLI(object):

    def __init__(self):
        self._model = ArchiClassifierModel()
       
    def train(self, dataset):
        return self._model.train(dataset)
    
    def predict(self,dataset):
        return self._model.predict(dataset)
    
if __name__ == '__main__':
    logging.basicConfig(filename='myapp.log',level=logging.INFO)
    fire.Fire(CLI)