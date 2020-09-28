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
FEATURES = 12

def evaluate_predictions(model, data):
    total = len(data)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    correct_prob = 0
    incorrect_prob = 0
    pyr_prob = 0
    in_prob = 0
    #correct_waveforms = 0
    correct_clusters = 0
    for cluster in data:
        input, label = NN_util.parse_test(cluster)
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        prediction, prob = model.predict(input)
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

        if prediction == label:
            correct_prob += prob
        else:
            incorrect_prob += prob
        if label == 1:
            pyr_prob += prob
        else:
            in_prob += prob
    correct_prob = correct_prob / correct_clusters
    incorrect_prob = incorrect_prob / (total - correct_clusters)
    pyr_prob = pyr_prob / total_pyr
    in_prob = in_prob / total_in
        
    print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, 100 * correct_clusters / total, '%'))
    print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
    print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    print('mean confidence in correct clusters is %.4f and in incorrect clusters is %.4f' % (correct_prob, incorrect_prob))
    print('mean confidence for pyramidal is %.4f and for interneurons is %.4f' % (pyr_prob, in_prob))
    return correct_clusters, correct_clusters / total
        

def run():
    print('Reading data...')
    data = NN_util.read_data('clustersData', should_filter = True)
    print('Splitting data...')
    train, dev, test = NN_util.split_data(data)
    train = NN_util.squeeze_clusters(train)
    dev = NN_util.squeeze_clusters(dev)

    """ this should be the same as the following calculation of the ratio
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

    trainer = SupervisedTrainer(criterion = criterion, batch_size = BATCH_SIZE)

    model = Net(32, 64, FEATURES, CLASSES)

    print('Starting training...')
    best_epoch = trainer.train(model, train, num_epochs = EPOCHS, dev_data = dev, learning_rate = LR)

    print('best epoch was %d' % (best_epoch))
    
    trainer.load_model(best_epoch, model)

    evaluate_predictions(model, test)

if __name__ == "__main__":
    run()
