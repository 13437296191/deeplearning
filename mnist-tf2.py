import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import struct
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

def load_mnist(path, kind='train'):
    """作用：从`path`中读取数据集
    ·path：表示数据集的路径
    ·kind：表示要读取那一部分的数据，有'test'与'train'可选,默认为'train'
    """
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    imshape = images.shape
    images = images.reshape(imshape[0],28,28)
 
    return images, labels
#def show_mnist(images,labels):
#    for i in range(25):
 #       plt.subplot(5,5,i+1)
  #     plt.yticks([ ])
   #     plt.grid(False)
     #   plt.imshow(images[i],cmap=plt.cm.gray)
      #  plt.xlabel(str(labels[i]))
   # plt.show()
 
def one_hot(labels):
    onehot_labels=np.zeros(shape=[len(labels),10])
    for i in range(len(labels)):
        index=labels[i]
        onehot_labels[i][index]=1
    return onehot_labels
 
def mnist_net():
    model = Sequential()
    model.add(Conv2D(64,(3,3), activation='relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model
 
 
def trian_model(train_images,train_labels,test_images,test_labels):
    # re-scale to 0~1.0之间
    train_images=train_images/255.0
    test_images=test_images/255.0
    train_images = train_images.reshape(-1,28,28,1)
    test_images = test_images.reshape(-1,28,28,1)
    train_labels=one_hot(train_labels)
    test_labels=one_hot(test_labels)
 
    # 建立模型
    model = mnist_net()
    
    model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
    model.fit(train_images,train_labels,batch_size=32,epochs=5)
 
    test_loss,test_acc=model.evaluate(test_images,test_labels,batch_size=32)
    print("Test Accuracy %.2f"%test_acc)
    model.summary()
 
    # 开始预测
  #  cnt=0
  #  predictions=model.predict(test_images)
  #  for i in range(len(test_images)):
  #      target=np.argmax(predictions[i])
  #      label=np.argmax(test_labels[i])
  #      if target==label:
  #          cnt +=1
  #  print("correct prediction of total : %.2f"%(cnt/len(test_images)))
# 
  #  model.save('mnist-model.h5')
 
if __name__=="__main__":

    train = load_mnist('C:\\Users\\Administrator\\mnist\\datasets')
    train_images, train_labels = train
    test = load_mnist('C:\\Users\\Administrator\\mnist\\datasets',kind='t10k')
    test_images, test_labels = test
   # show_mnist(train_images, train_labels)
    trian_model(train_images, train_labels, test_images, test_labels)
