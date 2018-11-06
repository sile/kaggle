import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(7, 9)
            self.l2 = L.Linear(9, 3)
            self.l3 = L.Linear(3, 7)
            self.l4 = L.Linear(7, 2)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)
        return y

def make_model():
    return L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
