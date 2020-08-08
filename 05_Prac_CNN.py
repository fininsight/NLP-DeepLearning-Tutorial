import numpy as np

def ReLU() :
    pass

def convolution(images, n_F, F, S, P=0) :
  weights = [np.random.randn(F[0], F[1]) for i in range(n_F)]

  h = int((images[0].shape[0]-F[0]+2*P)/S[0]+1)
  w = int((images[0].shape[1]-F[1]+2*P)/S[1]+1)

  map = np.zeros((n_F, h, w))

  #(W−F+2P)/S+1
  for i, weight in enumerate(weights) :
    for img in images :
      for y in range(0, h, S[0]) :
        for x in range(0, w, S[1]) :
          tmp = img[y:y+F[0], x:x+F[1]][:,:,0]
          map[i, y, x] = np.sum(np.multiply(weight, tmp))
  return map

def maxpooling(map, pooling) :
  h = int(map[0].shape[0]/pooling[0])
  w = int(map[0].shape[1]/pooling[1])

  mat = np.zeros((map.shape[0], h, w))

  for i in range(map.shape[0]) :
    for y in range(0, h, pooling[0]) :
      for x in range(0, w, pooling[1]) :
        #print(map[i,y:y+pooling[0],x:x+pooling[1]])
        mat[i, y, x] = np.max(map[i,y:y+pooling[0],x:x+pooling[1]])

  return mat

def fullyconected() :
  pass


def CNN(data, label) :
    pass
  #convolution()


if __name__ == "__main__" :
    from tensorflow.keras import datasets

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)) #데이터 건수, 이미지 높이, 이미지 너비, 컬러 채널
    test_images = test_images.reshape((10000, 28, 28, 1)) #데이터 건수, 이미지 높이, 이미지 너비, 컬러 채널

    # 픽셀 값을 0~1 사이로 정규화합니다.
    train_images, test_images = train_images / 255.0, test_images / 255.0

    conv1 = convolution([train_images[:2]], n_F=32, F=(3, 3), S=(1, 1))
    pooling1 = maxpooling(conv1, (2,2))

    conv2 = convolution(conv1, n_F=64, F=(3, 3), S=(1, 1))
    pooling2 = maxpooling(conv2, (2,2))