import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord("datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/records100/00000/00001_lr")

signal = record.p_signal

print("Shape:", signal.shape)

plt.plot(signal[:,0])
plt.title("ECG Lead 1")
plt.show()