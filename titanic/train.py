import model
import numpy as np
import chainer
import csv
from chainer import training
from chainer.training import extensions

#
# Parameters
#
epoch = 50
batchsize = 4

#
# Data
#
def embarked_to_int(embarked):
    if embarked == 'C':
        return 0
    elif embarked == 'S':
        return 1
    elif embarked == 'Q':
        return 2
    else:
        return 3

trainx = []
trainy = []
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        trainy.append(row[1]) # Surviced
        trainx.append([
            row[2], # Pclass
            row[4] == 'female', # Sex
            row[5] or 0, # Age
            row[6], # SibSp
            row[7], # Parch
            row[9], # Fare
            embarked_to_int(row[11]) # Embarked
        ])

trainx = np.array(trainx, dtype=np.float32)
trainy = np.array(trainy, dtype=np.int32)
train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

#
# Optimizer
#
model = model.make_model()
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

#
# Iterators
#
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

#
# Updater
#
updater = training.StandardUpdater(train_iter, optimizer)

#
# Trainer
#
trainer = training.Trainer(updater, (epoch, 'epoch'))

#
# Extensions
#
trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

#
# Run
#
trainer.run()
chainer.serializers.save_npz('result/out.model', model)
