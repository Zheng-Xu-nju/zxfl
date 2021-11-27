import numpy as np
import matplotlib.pyplot as plt

Accuracy_list = [2]
x1 = range(0, 1)
y1 = Accuracy_list
plt.subplot(1, 1, 1)  #plt.subplot('行','列','编号')
plt.plot(x1, y1, '.-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.show()
plt.savefig("accuracy_loss.jpg")