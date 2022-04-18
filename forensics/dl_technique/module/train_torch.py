import torch
import numpy as np
import random
import cv2
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torchvision
import torchsummary
from torch.optim import Adam
from torch import optim
import torch.backends.cudnn as cudnn

from sklearn import metrics
from sklearn.metrics import recall_score,accuracy_score,precision_score,log_loss,classification_report

from utils.Log import Logger
from dataloader.gen_dataloader import *

import sys, os
import os.path as osp
sys.path.append(osp.dirname(__file__))

from loss.focal_loss import FocalLoss as FL
from loss.weightedBCE_loss import WeightedBinaryCrossEntropy as WBCE

def save_result(text_writer, log, iteration, train_loss, train_acc, val_loss, val_mean_acc, val_acc, pre, rec, f1, is_epoch=True):
    # print result
    if is_epoch:
        result = "Epoch {} --- ".format(iteration)
    else:
        result = "Iter {} --- ".format(iteration)  
    result += "Train loss: {:.4f} --- Train accuracy: {:.4f} --- Valid loss: {:.4f} --- Valid mean accuracy: {:.4f} --- Valid accuracy score: {:.4f} --- Valid precision: {:.4f} --- Valid recall: {:.4f} --- Valid F1-Score: {:.4f}".format(train_loss, train_acc, val_loss, val_mean_acc, val_acc, pre, rec, f1)
    print(result)
    
    # Save result per epoch
    text_writer.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(iteration, train_loss, train_acc, val_loss, val_acc, pre, rec, f1))
    if is_epoch:
        scalar_dict_loss = {
            "Train loss/Epoch": train_loss, 
            "Validation loss/Epoch": val_loss
        }
        scalar_dict_accuracy = {
            "Train accuracy/Epoch": train_acc,
            "Validation accuracy/Epoch": val_mean_acc
        }
        
        scalar_dict_metric = {
            "Accuracy/Epoch": val_acc,
            "Precision/Epoch": pre,
            "Recall/Epoch": rec,
            "F1-Score/Epoch": f1
        }
    else:
        scalar_dict_loss = {
            "Train loss/Step": train_loss, 
            "Validation loss/Step": val_loss
        }
        scalar_dict_accuracy = {
            "Train accuracy/Step": train_acc,
            "Validation accuracy/Step": val_mean_acc
        }
        
        scalar_dict_metric = {
            "Accuracy/Step": val_acc,
            "Precision/Step": pre,
            "Recall/Step": rec,
            "F1-Score/Step": f1
        }
        
    log.write_scalar(scalar_dict=scalar_dict_loss, global_step=iteration)
    log.write_scalar(scalar_dict=scalar_dict_accuracy, global_step=iteration)
    log.write_scalar(scalar_dict=scalar_dict_metric, global_step=iteration)

#############################################
################# SINGLE SPATIAL IMAGE STREAM
#############################################

def eval_image_stream(model ,dataloader_val,device,criterion,adj_brightness=1.0, adj_contrast=1.0 ):
    val_loss = 0
    val_accuracy = 0
    model.eval()
    y_label = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_val):
            # Push to device
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs)
            logps = logps.squeeze()

            # Loss in a batch
            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            val_loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    val_loss /= len(dataloader_val)
    val_accuracy /= len(dataloader_val)
    accuracy = accuracy_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    f1 = 2.0 * recall * precision / (recall + precision)
    # print(classification_report(y_label,y_pred_label))
    model.train()
    return val_loss, val_accuracy, accuracy, precision, recall, f1

def train_image_stream(model, criterion_name=None, train_dir = '', val_dir ='', image_size=256, lr=3e-4, \
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=20, eval_per_iters=-1, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="xception", args_txt=""):

    # Early stopping epochs
    patience = es_patience
    best_loss = sys.float_info.max
    best_acc = 0.0

    # Vari
    save_loss = sys.float_info.max
    save_acc = 0.0

    # Create checkpoint dir and sub-checkpoint dir for each hyperparameter:
    if not osp.exists(checkpoint):
        os.makedirs(checkpoint)
    ckc_pointdir = osp.join(checkpoint, args_txt)
    if not osp.exists(ckc_pointdir):
        os.makedirs(ckc_pointdir)
    # Save log with tensorboard
    log = Logger(os.path.join(ckc_pointdir, "logs"))
    
    # Define devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True
    
    # Define and load model
    model = model.to(device)
    if resume != "":
        model.load_state_dict(torch.load(osp.join(ckc_pointdir, resume)))

    # Define optimizer (Adam) and learning rate decay
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3*i for i in range(1, epochs//3 + 1)], gamma = 0.8)

    # Generate dataloader train and validation 
    dataloader_train, dataloader_val, num_samples = generate_dataloader_image_stream(train_dir, val_dir, image_size, batch_size, num_workers)
    
    # Define criterion
    criterion = None
    if criterion_name[0] == 'bce':
        criterion = nn.BCELoss()
        print("Use binary cross entropy loss.")
    else:
        if criterion_name[0] == 'wbce':
            weight_class_0 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[0])
            weight_class_1 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[1])
            weights = [weight_class_0, weight_class_1]
            criterion = WBCE(weights=weights)
            print("Use weighted cross entropy loss: ", weights)
        elif criterion_name[0] == 'focalloss':
            weight_class_0 = num_samples[1] / (num_samples[0] + num_samples[1])
            weight_class_1 = 1 - weight_class_0
            weights = [weight_class_0, weight_class_1]
            criterion = FL(weights=weights, gamma=float(criterion_name[1]))
            print("Use focal loss: ", weights, "gamma: ", float(criterion_name[1]))
    criterion = criterion.to(device)

    # Writer instance for <epoch||train loss||train acc||val loss||val acc>
    text_writer_epoch = open(osp.join(ckc_pointdir, 'train_epoch.csv'), 'w')
    text_writer_epoch.write("Epoch, Train loss, Train accuracy, Validation loss, Validation accuracy, Validation precision, Validation recall, Validation F1-Score\n")
    # Writer instance for <iter||loss per batch>
    text_writer_single_step = open(osp.join(ckc_pointdir, 'train_single_step.csv'), 'w')
    text_writer_single_step.write("Iter, Loss per batch\n")
    # Writer instance for <eval_per_iters>: <iter||train loss||train acc||val loss||val acc>
    text_writer_multi_step = open(osp.join(ckc_pointdir, 'train_multi_step.csv'), 'w')
    text_writer_multi_step.write("Iter, Train loss, Train accuracy, Validation loss, Validation accuracy, Validation precision, Validation recall, Validation F1-Score\n")

    # Save model to txt file
    sys.stdout = open(os.path.join(checkpoint, 'model_{}.txt'.format(args_txt)), 'w')
    torchsummary.summary(model, (3, image_size, image_size))
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    model.train()

    running_loss = 0
    running_acc = 0

    global_loss = 0.0
    global_acc = 0.0
    global_step = 0
    begin_time = time.time()

    for epoch in range(epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("lr = ", optimizer.param_groups[0]['lr'])

        # Train
        model.train()
        print("Training...")
        for inputs, labels in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            logps = model.forward(inputs)   # Shape (32, 1)
            logps = logps.squeeze()         # Shape (32, )

            # Find loss
            loss = criterion(logps, labels)
            
            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            running_loss += loss.item()
            global_loss += loss.item()
            equals = (labels == (logps > 0.5))
            running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            global_acc += torch.mean(equals.type(torch.FloatTensor)).item()

            # Save step's loss:
            # To tensorboard and to writer
            scalar_dict = {"Loss/train_single_step": loss.item()}
            log.write_scalar(scalar_dict=scalar_dict, global_step=global_step)
            text_writer_single_step.write("{},{:.4f}\n".format(global_step, loss.item()))

            # Eval after <?> iters:
            if eval_per_iters != -1:
                if global_step % eval_per_iters == 0:
                    # Eval
                    val_loss, val_mean_acc, val_acc, pre, rec, f1 = eval_image_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    # Save txt and logger:
                    save_result(text_writer_multi_step, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mean_acc, val_acc, pre, rec, f1, is_epoch=False)
                    model.train()

        running_loss /= len(dataloader_train)
        running_acc /= len(dataloader_train)

        # Eval
        print("Validating...")
        val_loss, val_mean_acc, val_acc, pre, rec, f1 = eval_image_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        save_result(text_writer_epoch, log, epoch+1, running_loss, running_acc, val_loss, val_mean_acc, val_acc, pre, rec, f1)

        # Reset to the next epoch
        running_loss = 0
        running_acc = 0
        scheduler.step()

        # Save last model
        torch.save(model.state_dict(), os.path.join(ckc_pointdir, 'model_last.pt'))

        # Save best loss and best accuracy model:
        if val_loss <= save_loss:
            save_loss = val_loss
            torch.save(model.state_dict(), osp.join(ckc_pointdir, 'model_best_loss.pt'))
        if val_acc >= save_acc:
            save_acc = val_acc
            torch.save(model.state_dict(), osp.join(ckc_pointdir, 'model_best_acc.pt'))

        torch.save(model.state_dict(), osp.join(ckc_pointdir, "epoch_{}.pt".format(epoch+1)))
        # Early stopping:
        if es_metric == "val_loss":
            if val_loss <= best_loss:
                best_loss = val_loss
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                print("### BEST LOSS ### at epoch  ", epoch+1, 'with loss {} and accuracy {}'.format(best_loss, val_acc))
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best val loss and val accuracy: {:.3f}, {:.3f}'.format(best_loss, val_acc))
                    break
        elif es_metric == "val_acc":
            if val_acc >= best_acc:
                best_acc = val_acc
                patience = es_patience
                print("### BEST ACCURACY ### at epoch  ", epoch+1, 'with loss {} and accuracy {}'.format(val_loss, best_acc))
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best val loss and val accuracy: {:.3f}, {:.3f}'.format(val_loss, best_acc))
                    break
    return

#############################################
################# DUAL STREAM
#############################################
def eval_dual_stream(model, dataloader_val,device,criterion,adj_brightness=1.0, adj_contrast=1.0 ):
    val_loss = 0
    val_accuracy = 0
    model.eval()
    # Find other metrics
    y_label = []
    y_pred_label = []
    with torch.no_grad():
        for inputs, fft_imgs, labels in dataloader_val:
            y_label.extend(labels.cpu().numpy().astype(np.float64))
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)

            # Forward network
            logps = model.forward(inputs, fft_imgs)
            logps = logps.squeeze()

            # Loss in a batch
            batch_loss = criterion(logps, labels)
            # Cumulate into running val loss
            val_loss += batch_loss.item()

            # Find accuracy
            equals = (labels == (logps > 0.5))
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            #
            logps_cpu = logps.cpu().numpy()
            pred_label = (logps_cpu > 0.5)
            y_pred_label.extend(pred_label)

    val_loss /= len(dataloader_val)
    val_accuracy /= len(dataloader_val)
    accuracy = accuracy_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    f1 = 2.0 * recall * precision / (recall + precision)
    # print(classification_report(y_label,y_pred_label))
    return val_loss, val_accuracy, accuracy, precision, recall, f1

def train_dual_stream(model, criterion_name=None, train_dir = '', val_dir ='', image_size=256, lr=3e-4, \
              batch_size=16, num_workers=8, checkpoint='', resume='', epochs=20, eval_per_iters=-1, \
              adj_brightness=1.0, adj_contrast=1.0, es_metric='val_loss', es_patience=5, model_name="dual-efficient", args_txt=""):

    # Early stopping epochs
    patience = es_patience
    best_loss = sys.float_info.max
    best_acc = 0.0

    # Vari
    save_loss = sys.float_info.max
    save_acc = 0.0

    # Create checkpoint dir and sub-checkpoint dir for each hyperparameter:
    if not osp.exists(checkpoint):
        os.makedirs(checkpoint)
    ckc_pointdir = osp.join(checkpoint, args_txt)
    if not osp.exists(ckc_pointdir):
        os.makedirs(ckc_pointdir)
    # Save log with tensorboard
    log = Logger(os.path.join(ckc_pointdir, "logs"))

    # Define devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True
    
    # Define and load model
    model = model.to(device)
    if resume != "":
        model.load_state_dict(torch.load(osp.join(checkpoint, resume)))

    # Define optimizer (Adam) and learning rate decay
    init_lr = lr
    init_epoch = 0
    if resume != "":
        try:
            init_epoch = int(resume.split('_')[1])
            init_lr = lr * (0.8 ** ((init_epoch - 1) // 3))
        except:
            pass
    print("Init epoch: ", init_epoch)
    print("Init lr: ", init_lr)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3*i for i in range(1, epochs//3 + 1)], gamma = 0.8)

    # Generate dataloader train and validation 
    dataloader_train, dataloader_val, num_samples = generate_dataloader_dual_stream(train_dir, val_dir, image_size, batch_size, num_workers)

    # Define criterion
    criterion = None
    if criterion_name[0] == 'bce':
        criterion = nn.BCELoss()
        print("Use binary cross entropy loss.")
    else:
        weight_class_0 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[0])
        weight_class_1 = 0.5 * (num_samples[0] + num_samples[1]) / (num_samples[1])
        weights = [weight_class_0, weight_class_1]
        if criterion_name[0] == 'wbce':
            criterion = WBCE(weights=weights)
            print("Use weighted cross entropy loss: ", weights)
        elif criterion_name[0] == 'focalloss':
            criterion = FL(weights=weights, gamma=float(criterion_name[1]))
            print("Use focal loss: ", weights, "gamma: ", float(criterion_name[1]))
    criterion = criterion.to(device)
    
    # Writer instance for <epoch||train loss||train acc||val loss||val acc>
    text_writer_epoch = open(osp.join(ckc_pointdir, 'train_epoch.csv'), 'a')
    text_writer_epoch.write("Epoch, Train loss, Train accuracy, Validation loss, Validation accuracy, Validation precision, Validation recall, Validation F1-Score\n")
    # Writer instance for <iter||loss per batch>
    text_writer_single_step = open(osp.join(ckc_pointdir, 'train_single_step.csv'), 'a')
    text_writer_single_step.write("Iter, Loss per batch\n")
    # Writer instance for <eval_per_iters>: <iter||train loss||train acc||val loss||val acc>
    text_writer_multi_step = open(osp.join(ckc_pointdir, 'train_multi_step.csv'), 'a')
    text_writer_multi_step.write("Iter, Train loss, Train accuracy, Validation loss, Validation accuracy, Validation precision, Validation recall, Validation F1-Score\n")

    model.train()

    running_loss = 0
    running_acc = 0

    global_loss = 0.0
    global_acc = 0.0
    global_step = 0

    for epoch in range(init_epoch, epochs):
        print("\n=========================================")
        print("Epoch: {}/{}".format(epoch+1, epochs))
        print("Model: {} - {}".format(model_name, args_txt))
        print("lr = ", optimizer.param_groups[0]['lr'])

        # Train
        model.train()
        print("Training...")
        for inputs, fft_imgs, labels in tqdm(dataloader_train):
            global_step += 1
            # Push to device
            inputs, fft_imgs, labels = inputs.float().to(device), fft_imgs.float().to(device), labels.float().to(device)
            # Clear gradient after a step
            optimizer.zero_grad()

            # Forward netword
            logps = model.forward(inputs, fft_imgs)     # Shape (32, 1)
            logps = logps.squeeze()                     # Shape (32, )

            # Find loss
            loss = criterion(logps, labels)
            
            # Backpropagation and update weights
            loss.backward()
            optimizer.step()

            # update running (train) loss and accuracy
            running_loss += loss.item()
            global_loss += loss.item()
            equals = (labels == (logps > 0.5))
            running_acc += torch.mean(equals.type(torch.FloatTensor)).item()
            global_acc += torch.mean(equals.type(torch.FloatTensor)).item()

            # Save step's loss:
            # To tensorboard and to writer
            scalar_dict = {"Loss/train_single_step": loss.item()}
            log.write_scalar(scalar_dict=scalar_dict, global_step=global_step)
            text_writer_single_step.write("{},{:.4f}\n".format(global_step, loss.item()))

            # Eval after <?> iters:
            if eval_per_iters != -1:
                if global_step % eval_per_iters == 0:
                    # Eval
                    # print("Validating...")
                    model.eval()
                    val_loss, val_mean_acc, acc, pre, rec, f1 = eval_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_brightness)
                    # Save txt and logger:
                    save_result(text_writer_multi_step, log, global_step, global_loss/global_step, global_acc/global_step, val_loss, val_mean_acc, acc, pre, rec, f1, is_epoch=False)
                    model.train()

        # Eval
        print("Validating...")
        model.eval()
        val_loss, val_mean_acc, val_acc, pre, rec, f1 = eval_dual_stream(model, dataloader_val, device, criterion, adj_brightness=adj_brightness, adj_contrast=adj_contrast)
        save_result(text_writer_epoch, log, epoch+1, running_loss/len(dataloader_train), running_acc/len(dataloader_train), val_loss, val_mean_acc, val_acc, pre, rec, f1)

        # Reset to the next epoch
        running_loss = 0
        running_acc = 0
        scheduler.step()

        # Save last model
        torch.save(model.state_dict(), os.path.join(ckc_pointdir, 'model_last.pt'))

        # Save best loss and best accuracy model:
        if val_loss <= save_loss:
            save_loss = val_loss
            torch.save(model.state_dict(), osp.join(ckc_pointdir, 'model_best_loss.pt'))
        if val_acc >= save_acc:
            save_acc = val_acc
            torch.save(model.state_dict(), osp.join(ckc_pointdir, 'model_best_acc.pt'))

        torch.save(model.state_dict(), osp.join(ckc_pointdir, "epoch_{}_{}_{}.pt".format(epoch+1, val_loss, val_acc)))
        # Early stopping:
        if es_metric == "val_loss":
            if val_loss <= best_loss:
                best_loss = val_loss
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                print("### BEST LOSS ### at epoch  ", epoch+1, 'with loss {} and accuracy {}'.format(best_loss, val_acc))
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best val loss and val accuracy: {:.4f}, {:.4f}'.format(best_loss, val_acc))
                    break
        elif es_metric == "val_acc":
            if val_acc >= best_acc:
                best_acc = val_acc
                patience = es_patience
                print("### BEST ACCURACY ### at epoch  ", epoch+1, 'with loss {} and accuracy {}'.format(val_loss, best_acc))
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best val loss and val accuracy: {:.4f}, {:.4f}'.format(val_loss, best_acc))
                    break
    return



