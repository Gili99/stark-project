import torch
import torch.nn as nn
import NN_util
from NN_model import Net
from NN_trainer import SupervisedTrainer
import numpy as np

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
CLASSES = 2
FEATURES = 8

def evaluate_predictions(model, data):
    total = len(data)
    #correct_waveforms = 0 might be interesting
    correct_clusters = 0
    for cluster in data:
        input, label = NN_util.parse_test(cluster)
        prediction, prob = model.predict(input)
        correct_clusters += 1 if prediction == label else 0
    print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, correct_clusters / total, '%'))
    return correct_clusters, correct_clusters / total
        

def run():
    data = NN_util.read_data('clustersData', should_filter = True)
    train, dev, test = NN_util.split_data(data)
    train = NN_util.squeeze_clusters(train)
    dev = NN_util.squeeze_clusters(dev)

    """ thisshould be the same as the following calculation of the ratio
    total = 0
    one_label = 0
    for sample in train:
        if sample[-1] == 1:
            one_label += 1
        total += 1
    ratio_a = one_label / total
    """

    one_label = train[train[:,-1] == 1]
    #ratio of pyramidal waveforms
    #note that it is over all waveforms and not specifically over clusters (each cluster can have different number of waveforms)
    ratio = one_label.shape[0] / train.shape[0] 

    #assert ratio == ratio_a #sanity check
            
    class_weights = torch.tensor([ratio/(1-ratio), 1.0]) #crucial as there is overrepresentation of pyramidal neurons in the data
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    trainer = SupervisedTrainer(criterion =criterion, batch_size = BATCH_SIZE)

    model = Net(32, 64, FEATURES, CLASSES)

    model = trainer.train(model, train, num_epochs = EPOCHS, dev_data = dev, learning_rate = LR)

    evaluate_predictions(model, test)

if __name__ == "__main__":
    run()
