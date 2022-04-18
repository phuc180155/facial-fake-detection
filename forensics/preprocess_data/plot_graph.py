import numpy as np
import matplotlib.pyplot as plt

import csv

csv_file = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/preprocess_data/auxiliary/efficient_vit_res.csv"
file = open(csv_file)
csvreader = csv.reader(file)

epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []
val_pre = []
val_rec = []
val_f1 = []

for row in csvreader:
    epochs.append(int(row[0]))
    train_loss.append(float(row[1]))
    train_acc.append(float(row[2]))
    val_loss.append(float(row[3]))
    val_acc.append(float(row[4]))
    val_pre.append(float(row[5]))
    val_rec.append(float(row[6]))
    val_f1.append(float(row[7]))


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)

plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.legend()
plt.title("Loss")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.1*i for i in range(1, 16)])

plt.subplot(1, 2, 2)

plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.legend()
plt.title("Accuracy")
plt.xticks([i for i in range(1, len(epochs)+1)])
# plt.yticks([0.9 + 0.01*i for i in range(0, 11)])

plt.savefig('myplot.png')

# plt.subplot(2, 1, 2)
# # plt.imshow([])
# plt.title("Loss")