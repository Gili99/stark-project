import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from NN_util import create_batches
from NN_evaluator import Evaluator

import logging
import os
import random
import time
from tqdm import tqdm

class SupervisedTrainer(object):
   """ The SupervisedTrainer class helps in setting up a training framework in a
   supervised setting.

   Args:
      criterion (optional): loss for training, (default: CrossEntropyLoss)
      batch_size (int, optional): batch size for experiment, (default: 32)
      print_every (int, optional): number of batches to print after, (default: 100)
   """
   def __init__(self, criterion = nn.CrossEntropyLoss(), batch_size = 32, random_seed = None, print_every = 100,
                eval_criterion = nn.CrossEntropyLoss()):
      self._trainer = "Simple Trainer"
      self.random_seed = random_seed
      if random_seed is not None:
         random.seed(random_seed)
         torch.manual_seed(random_seed)
      self.criterion = criterion
      self.evaluator = Evaluator(criterion = eval_criterion, batch_size = batch_size)
      self.optimizer = None
      self.print_every = print_every

      self.batch_size = batch_size

      self.logger = logging.getLogger(__name__)

   def _train_batch(self, input_var, labels, model):
      # Forward propagation
      outputs = model(input_var)
        
      # Get loss
      criterion = self.criterion
      # zero the parameter gradients
      self.optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(input_var)
      loss = criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()

      return loss.item()

   def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, dev_data = None):
      log = self.logger

      print_loss_total = 0  # Reset every print_every
      epoch_loss_total = 0  # Reset every epoch

      device = torch.device('cuda') if torch.cuda.is_available() else -1

      steps_per_epoch = len(data) // self.batch_size + (1 if len(data) % self.batch_size != 0 else 0)
      total_steps = steps_per_epoch * n_epochs

      step = start_step
      for epoch in tqdm(range(start_epoch, n_epochs + 1)):
         log.debug("Epoch: %d, Step: %d" % (epoch, step))

         #create batches
         batches = create_batches(data, self.batch_size)
            
         for batch in batches:
            step += 1

            input_var, labels = batch #need to make sure it is indeed var

            loss = self._train_batch(input_var, labels, model)

            # Record average loss
            print_loss_total += loss
            epoch_loss_total += loss

            if step == 0: continue

            if step % self.print_every == 0:
               print_loss_avg = print_loss_total / self.print_every
               print_loss_total = 0
               log_msg = 'Progress: %d%%, Train: %.4f' % (step / total_steps * 100, print_loss_avg)
               log.info(log_msg)

         epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
         epoch_loss_total = 0
         log_msg = "Finished epoch %d: Train: %.4f" % (epoch, epoch_loss_avg)
         if dev_data is not None:
            dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
            #here we should update lr if we will ahve scheduler or just according to the epoch loss if we won't have dev
            log_msg += ", Dev: %.4f, Accuracy: %.4f" % (dev_loss, accuracy)
            print(log_msg)
            model.train(mode = True)

         log.info(log_msg)

   def train(self, model, data, num_epochs = 10, dev_data = None, optimizer = None, learning_rate = 0.001):
      """ Run training for a given model.

      Args:
         model: model to run training on
         data: dataset object to train on
         num_epochs (int, optional): number of epochs to run (default 10)
         dev_data (optional): dev Dataset (default None)
         optimizer (optional): optimizer for training (default: Optimizer(pytorch.optim.SGD))
      Returns:
         model: trained model.
      """

      start_epoch = 1
      step = 0
      if optimizer is None:
         optimizer =  optim.SGD(model.parameters(), lr = learning_rate)
         self.optimizer = optimizer

        #think of adding a scheduler

         self.logger.info("Optimizer: %s" % (self.optimizer))

         self._train_epoches(data, model, num_epochs, start_epoch, step, dev_data = dev_data)
         return model

def run():
    pass                   
      
      


        
