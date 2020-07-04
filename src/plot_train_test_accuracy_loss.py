import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

x_values = []
y_values = []

index = count()
figure = plt.figure()

def animate(i):

	data_path = "/home/vibek/Anomanly_detection_packages/DNN_Package/"

	data = data_path+"training_set_dnnanalysis.csv"

	read_data = pd.read_csv(data)

	x_values = read_data['accuracy']
	y_values = range(len(read_data))
	#plt.clf()
	plt.plot(y_values, x_values)
	plt.pause(0.05)
	plt.title('Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.gcf().autofmt_xdate()
    #plt.tight_layout()

ani = FuncAnimation(figure, animate, interval=20)
plt.show()


# plot loss during training
#pyplot.subplot(211)
#pyplot.title('Loss')
#pyplot.xlabel('Epochs')
#pyplot.ylabel('Loss')
#pyplot.plot(read_data['loss'], label='train')
#pyplot.plot(read_data['val_loss'], label='test')
#pyplot.legend()
# plot accuracy during training
#pyplot.subplot(212)
#pyplot.title('Accuracy')
#pyplot.xlabel('Epochs')
#pyplot.ylabel('Accuracy')
#pyplot.plot(read_data['acc'], label='train')
#pyplot.plot(read_data['val_acc'], label='test')
#pyplot.legend()
#pyplot.savefig("/home/vibek/Anomanly_detection_packages/DNN_Package/Training_Testing_result/loss_accuracy.png", dpi=300)
#pyplot.show()
