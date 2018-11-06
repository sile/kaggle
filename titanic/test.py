import model
import numpy as np
import chainer
import chainer.functions as F
import csv
from chainer import training
from chainer.training import extensions

#
# Parameters
#
epoch = 100
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

test = []
pids = []
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        pids.append(row[0])
        test.append([
            row[1], # Pclass
            row[3] == 'female', # Sex
            row[4] or 0, # Age
            row[5], # SibSp
            row[6], # Parch
            row[8] or 0, # Fare
            embarked_to_int(row[10]) # Embarked
        ])

test = np.array(test, dtype=np.float32)

model = model.make_model()
chainer.serializers.load_npz('result/out.model', model)

print('PassengerId,Survived')
for i in range(len(test)):
    x = chainer.Variable(test[i].reshape(1, 7))
    result = F.softmax(model.predictor(x))
    print('{},{}'.format(pids[i], result.data.argmax()))
