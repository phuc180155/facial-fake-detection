import numpy as np
import matplotlib.pyplot as plt

import csv

csv_file = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/result/train_loss_epoch.csv"
file = open(csv_file)
csvreader = csv.reader(file)

epochs = []
train_loss = []
test_loss = []

train_acc = []
test_acc = []

pre_0, pre_1, mic_pre, mac_pre = [], [], [], []
rec_0, rec_1, mic_rec, mac_rec = [], [], [], []
f1_0, f1_1, mic_f1, mac_f1 = [], [], [], []

# epoch, val_loss, acc, pre_0, rec_0, f1_0, pre_1, rec_1, f1_1, mic_pre, mic_rec, mic_f1, mac_pre, mac_rec, mac_f1
# epoch || train loss, val loss ||    train acc, test acc    ||   pre_0, rec_0, f1_0  ||   pre_1, rec_1, f1_1     ||   mic_pre, mic_rec, mic_f1     ||    mac_pre, mac_rec, mac_f1
for row in csvreader:
    epochs.append(int(row[0]))
    train_loss.append(float(row[1]))
    test_loss.append(float(row[2]))
    
    train_acc.append(float(row[3])) 
    test_acc.append(float(row[4]))

    pre_0.append(float(row[5]))
    rec_0.append(float(row[6]))
    f1_0.append(float(row[7]))
    
    pre_1.append(float(row[8]))
    rec_1.append(float(row[9]))
    f1_1.append(float(row[10]))
    
    mic_pre.append(float(row[11]))
    mic_rec.append(float(row[12]))
    mic_f1.append(float(row[13]))
    
    mac_pre.append(float(row[14]))
    mac_rec.append(float(row[15]))
    mac_f1.append(float(row[16]))

print(mic_pre)
print(mic_rec)
print(mic_f1)


########### LOSS
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, test_loss, label="Validation loss")
plt.legend()
plt.title("Loss")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])

########### METRIC
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, test_acc, label="Test accuracy")
plt.legend()
plt.title("Accuracy")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.9 + 0.01*i for i in range(0, 11)])
plt.savefig('loss.png')

plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
plt.plot(epochs, pre_0, label="Precision")
plt.plot(epochs, rec_0, label="Recall")
plt.plot(epochs, f1_0, label="F1-Score")
plt.legend()
plt.title("Class real's metric")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])

plt.subplot(1, 3, 2)
plt.plot(epochs, pre_1, label="Precision")
plt.plot(epochs, rec_1, label="Recall")
plt.plot(epochs, f1_1, label="F1-Score")
plt.legend()
plt.title("Class fake's metric")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])

# plt.subplot(2, 2, 3)
# plt.plot(epochs, mic_pre, label="Precision")
# plt.plot(epochs, mic_rec, label="Recall")
# plt.plot(epochs, mic_f1, label="F1-Score")
# plt.legend()
# plt.title("Micro Average Metric")
# plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])

plt.subplot(1, 3, 3)
plt.plot(epochs, mac_pre, label="Precision")
plt.plot(epochs, mac_rec, label="Recall")
plt.plot(epochs, mac_f1, label="F1-Score")
plt.legend()
plt.title("Macro Average Metric")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])
plt.savefig('metric.png')
# plt.subplot(2, 1, 2)
# # plt.imshow([])
# plt.title("Loss")