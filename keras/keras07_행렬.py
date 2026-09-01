import numpy as np

x1 = np.array([1,2,3]) #백터 데이터 (1차원) -> (3,) 

#무조건 먼저 데이터 구조 확인하기 .shape
print ("x1 = ",x1.shape)

x2 = np.array([[1,2,3]]) # (1,3) / 1x3
print ("x2 = ",x2.shape)

x3 = np.array([[1,2], [3,4]]) 
print ("x3 = ",x3.shape) # x3 =  (2, 2)

x4 = np.array([[1,2],[3,4],[5,6]])
print ("x4 = ",x4.shape) # x4 =  (3, 2)

x5 = np.array ([[1,2],[3,4],[5,6,]]) #리스트든 벡터든 함수에서든  Ln 17, col 34 같은 , 넣어도 됨

x6 = np.array([[[1,2],[3,4],[5,6,]]])
print ("x6 = ",x6.shape) # x6 =  (1, 3, 2)

x7 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
print ("x7 = ",x7.shape) # x7 =  (2, 2, 2)

x8 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]])
print ("x8 = ",x8.shape) # x8 =  (1, 1, 1, 2, 5)

x9 = np.array([[[1,2,3]], [[4,5,6]]])
print ("x9 = ",x9.shape) # x9 =  (2, 1, 3)

x10 = np.array([[[[1]]], [[[2]]]])
print ("x10 = ",x10.shape) # x10 =  (2, 1, 1, 1)

# 어떤 데이터를 받던간데 .shape 를 찍어봐야 한다.
