from matplotlib import pyplot
from tensorflow.keras.datasets import fashion_mnist
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(trainX[i], cmap =pyplot.get_cmap('gray'))
pyplot.show()