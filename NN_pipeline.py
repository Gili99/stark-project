import torch
import torch.nn as nn
import NN_util
from NN_model import Net
from NN_trainer import SupervisedTrainer
import numpy as np

EPOCHS = 100 # using patience we will rarely go over 50 and commonly stay in the 10-20 area 
PATIENCE = 10
BATCH_SIZE = 32
LR = 0.001
CLASSES = 2
FEATURES = 12

def repredict(probs, thr = 0.75):
    """
    Repredict using only the predictions with confidence higher than thr
    probs (array of size (n, CLASSES)): Indicates the confidence for each class on a specific feature chunck
    thr (float, optional): The threshold for removing predictions
    """
    probs_up = probs[probs.max(dim = 1)[0] >= thr]
    probs_up = probs[probs.max(dim = 1)[0] <= thr + 0.1]
    if len(probs) == 0: #return regular prediction
        prob = torch.mean(probs, dim = 0)
        arg_max = torch.argmax(prob)
        return arg_max
    predictions = probs_up.argmax(dim = 1)
    return torch.argmax(predictions)

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
    correct_chunks = 0
    total_chunks = 0
    correct_clusters = 0
    for cluster in data:
        input, label = NN_util.parse_test(cluster)
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        prediction, prob, raw = model.predict(input)
        #if prob < 0.7: # this doesn't seem to improve so it is commented out 
        #    prediction = repredict(raw)
        all_predictions = raw.argmax(dim = 1)
        total_chunks += len(all_predictions)
        correct_chunks += len(all_predictions[all_predictions == label])
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
    print('Number of correct classified feature chunks is %d, which is %.4f%s' % (correct_chunks, 100 * correct_chunks / total_chunks, '%'))
    print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
    print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    print('mean confidence in correct clusters is %.4f and in incorrect clusters is %.4f' % (correct_prob, incorrect_prob))
    print('mean confidence for pyramidal is %.4f and for interneurons is %.4f' % (pyr_prob, in_prob))
    return correct_clusters, correct_clusters / total
        

def run(path_load = None):
    print('Reading data...')
    data = NN_util.read_data('clustersData', should_filter = True)
    print('Splitting data...')
    train, dev, test = NN_util.split_data(data, per_train = 0.6, per_dev = 0.2, per_test = 0.2)
    train_squeezed = NN_util.squeeze_clusters(train)
    dev_squeezed = NN_util.squeeze_clusters(dev)

    one_label = train_squeezed[train_squeezed[:,-1] == 1]
    #ratio of pyramidal waveforms
    #note that it is over all waveforms and not specifically over clusters (each cluster can have different number of waveforms)
    ratio = one_label.shape[0] / train_squeezed.shape[0] 
            
    class_weights = torch.tensor([ratio / (1 - ratio), 1.0]) #crucial as there is overrepresentation of pyramidal neurons in the data
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    one_label = dev_squeezed[dev_squeezed[:,-1] == 1]
    ratio = one_label.shape[0] / dev_squeezed.shape[0]
    class_weights = torch.tensor([ratio / (1 - ratio), 1.0])
    eval_criterion = nn.CrossEntropyLoss(weight = class_weights)

    trainer = SupervisedTrainer(criterion = criterion, batch_size = BATCH_SIZE, patience = PATIENCE, eval_criterion = eval_criterion)

    model = Net(32, 64, FEATURES, CLASSES)

    if path_load == None:
        print('Starting training...')
        best_epoch = trainer.train(model, train_squeezed, num_epochs = EPOCHS, dev_data = dev_squeezed, learning_rate = LR)

        print('best epoch was %d' % (best_epoch))
        trainer.load_model(model, epoch = best_epoch)
    else:
        print('Loading model...')
        trainer.load_model(model, path = path_load)

    evaluate_predictions(model, test)

if __name__ == "__main__":
    #run(path_load = 'saved_models/epoch39')
    run()
