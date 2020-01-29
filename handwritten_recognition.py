import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from skimage.morphology import medial_axis, skeletonize
from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import label
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy import spatial
from math import sqrt

def open_file (folder_name, inplist, index):
	for filename in glob.glob(os.path.join(folder_name, '*jpg')):
		img = cv2.imread(filename)
		img = cv2.resize(img, (30,30))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		inplist[index].append(img)

def show_image (image):
	plt.figure()
	# Gray scale images, i.e. images with only one colorchannel, are interpreted by imshow as to be plotted using a colormap. 
	#Specify the colormap to one colorchannel (grayscale).
	plt.imshow(image, cmap = "gray", vmin = 0, vmax = 1)
	plt.show()

def plot_histogram(image):
	plt.hist(image.ravel(), 256, [0, 256]) 
	plt.show()

def calc_area(image, mlist):
	height, width = image.shape[:2]
	area = 0
	for i in range(0, height):
		for j in range(0 , width):
			if image[i][j] == 1 :
				area += 1
    			# area = counting all the pixels contained in the object
	mlist.append(area)

def find_edge(image, mlist):
	height, width = image.shape[:2]

	new_image = [[[] for col in range (0,width)] for row in range(0, height)]
	kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
	kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

	segmented_pixels = [[0,0,0],[0,0,0],[0,0,0]]

	perimeter = 0

	for i in range(0, height):
		for j in range(0 , width):
			# Find index neighbour
			segmented_pixels = neighbors(image, i, j, 1)

			res_x = matrix_multiplication(segmented_pixels, kernel_x)
			res_y = matrix_multiplication(segmented_pixels, kernel_y)

			val = int(sqrt((res_x[1][1]**2) + (res_y[1][1]**2)))

			new_image[i][j] = int(val/2)

			if new_image[i][j] >= 1:
				perimeter +=1

	# image = new_imag
	# show_image(new_image)
	mlist.append(perimeter)

def neighbors(mat, row, col, radius):
	rows, cols = len(mat), len(mat[0])
	out = []

	for i in range(row + 1 - radius - 1, row + 1 + radius):
		row = []
		for j in range(col + 1 - radius - 1, col + 1 + radius):
			if 0 <= i < rows and 0 <= j < cols:
				row.append(mat[i][j])
			else:
				row.append(0)
		out.append(row)
	return out

def matrix_multiplication(matrix_a, matrix_b):
	result = [[sum(a*b for a,b in zip(b_row, a_col)) for a_col in zip(*matrix_a)] for b_row in matrix_b]
	return result

def calc_compactness(area, perimeter, mlist):
	compactness = perimeter**2 / 4 * 3.14 * area
	mlist.append(compactness)

def count_no_of_holes(image, mlist):
	no_branches = count_intersection(image)
	no_endpoints = count_end_points(image)

	holes = 1 + ((no_branches - no_endpoints)/2)
	holes = int(holes)

	mlist.append(holes)

def count_intersection(image):
	height, width = image.shape[:2]
	skeleton_image = skeletonize(image).astype(np.uint8)
	# skeleton_image, distance = medial_axis(image, return_distance=True)
	intersections = list()
	neighbors_matrix = [[0,0,0],[0,0,0],[0,0,0]]

	validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]]
	
	for i in range(0, height):
		for j in range(0 , width):
			# If white pixel is found
			if skeleton_image[i][j] == 1:
				neighbors_matrix = neighbors(skeleton_image, i, j, 1)
				neighbors_matrix = np.array(neighbors_matrix)
				neighbors_matrix = neighbors_matrix.flatten()
				neighbors_matrix = np.delete(neighbors_matrix, 4).tolist()

				valid = True
				if neighbors_matrix in validIntersection:
					intersections.append((j,i))

	# Filter intersections to make sure we don't count them twice or ones that are very close together
	for point1 in intersections:
		for point2 in intersections:
			if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
				intersections.remove(point2)
	# Remove duplicates
	intersections = list(set(intersections))
	# print('no of intersections:', intersections)
	return len(intersections)

def count_end_points(image):
	height, width = image.shape[:2]
	skeleton_image = skeletonize(image).astype(np.uint8)
	neigh = [[0,0,0],[0,0,0],[0,0,0]]
	counter = 0
	for i in range (0, height):
		for j in range (0, width):
			if skeleton_image[i][j] == 1:
				neigh = neighbors(skeleton_image, i, j , 1)      
				if np.sum(neigh) == 2:
					print(i,j)
					counter +=1
	# print('no of endpoints: ', counter)
	return (counter)

def binarize_image (image, threshold_list):
	minv = np.amin(image)
	maxv = np.amax(image)
	height, width = image.shape[:2]

	# Using otsu method to find threshold
	# Histogram of color value
	ret,thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# thresh = np.array(thresh)
	# thresh = thresh.flatten()
	threshold_list.append(thresh)

	for i in range(0, height):
		for j in range(0 , width):
    		 if image[i][j] > ret:
    			 image[i][j] = 0
    		 else:
    			 image[i][j] = 1


def find_cosine_similarity(features_1, features_2, label, index):
	# 10 x 3
	row = len(features_2)
	similarity_per_row = [0.0 for i in range (0, row)]

	for r in range(0, row):
		similarity_per_row[r] = 1 - spatial.distance.cosine(features_1, features_2[r])
	
	# print('max similar', max(similarity_per_row))

	max_index = similarity_per_row.index(max(similarity_per_row))
	label[index] = max_index

def find_average(list):
	average = sum(list)/len(list)
	return average

##########################################################################################
#										OPEN FILE

train_path = "/Users/arnes/Desktop/Pattern_Recognition_Final_Assignment/Train/"
test_path = "/Users/arnes/Desktop/Pattern_Recognition_Final_Assignment/Test/"

train_images = [[] for i in range (0,10)]
test_images = [[] for i in range (0,10)]

labels_train = np.array([0]*30 + [1]*30 + [2]*30 + [3]*30 + [4]*30 + [5]*30 + [6]*30 + [7]*30 + [8]*30 + [9]*30)
labels_test = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5 + [8]*5 + [9]*5)

for i in range (0,10):
	path = train_path + str(i)
	path2 = test_path + str(i)
	open_file(path, train_images, i)
	open_file(path2, test_images, i)

train_images_threshold = [[] for i in range (0,10)]

train_images_area = [[] for i in range (0,10)]
train_images_perimeter = [[] for i in range (0,10)]
train_images_compactness = [[] for i in range (0,10)]
train_images_no_holes = [[] for i in range (0,10)]

for i in range (0, 10):
	for j in range (0, 30):
		binarize_image(train_images[i][j], train_images_threshold[i])
	# show_image(skeletonize(train_images[i][0]))


for i in range (0, 10):
	for j in range (0, 30):
		calc_area(train_images[i][j], train_images_area[i])
		find_edge(train_images[i][j], train_images_perimeter[i])
		calc_compactness(train_images_area[i][j], train_images_perimeter[i][j], train_images_compactness[i])
		count_no_of_holes(train_images[i][j], train_images_no_holes[i])
	

train_images_features = [[] for i in range (0,10)]

# AVERAGE OF TRAIN IMAGE FEATURES
for i in range (0, 10):
	train_images_features[i].append(find_average(train_images_no_holes[i]))
	train_images_features[i].append(find_average(train_images_area[i]))
	train_images_features[i].append(find_average(train_images_perimeter[i]))
	train_images_features[i].append(find_average(train_images_compactness[i]))

#####################################################################################
# 										TEST IMAGES

test_images_threshold = [[] for i in range (0,10)]
test_images_area = [[] for i in range (0,10)]
test_images_perimeter = [[] for i in range (0,10)]
test_images_compactness = [[] for i in range (0,10)]
test_images_features = [[] for i in range (0,10)]
test_images_no_holes = [[] for i in range (0,10)]

for i in range (0, 10):
	for j in range (0, 5):
		binarize_image(test_images[i][j], test_images_threshold[i])


for i in range (0, 10):
	for j in range (0, 5):
		calc_area(test_images[i][j], test_images_area[i])
		find_edge(test_images[i][j], test_images_perimeter[i])
		calc_compactness(test_images_area[i][j], test_images_perimeter[i][j], test_images_compactness[i])
		count_no_of_holes(test_images[i][j], test_images_no_holes[i])

for i in range (0, 10):
	for j in range (0, 5):
		test_images_features[i].append([test_images_no_holes[i][j], test_images_area[i][j],test_images_perimeter[i][j], test_images_compactness[i][j]])

###################################################################################
#							FEATURE MATCHING USING COSINE SIMILARITY

# print('train features',train_images_features[0])
# print('test features', test_images_features[0][0])

label_index = [[0] * len(test_images[0]) for i in range (0, len(test_images))]

for i in range (0,10):
	for j in range (0,5):
		find_cosine_similarity(test_images_features[i][j], train_images_features, label_index[i], j)

print('Similarity result: ', label_index)


###################################################################################
#									KNN CLASSIFIER


train_classes = [[] for i in range (0,10)]
test_classes = [[] for i in range (0,10)]

print('no of holes in training image: ', train_images_no_holes)
print('no of holes in testing image: ', train_images_no_holes)

for i in range (0, 10):
	train_classes [i] = list(zip(train_images_no_holes[i], train_images_area[i], train_images_perimeter[i], train_images_compactness[i]))
	test_classes [i] = list(zip(test_images_no_holes[i], test_images_area[i],test_images_perimeter[i], test_images_compactness[i]))

feature_set_train = np.vstack([train_classes[0], train_classes[1], train_classes[2], train_classes[3], train_classes[4], train_classes[5], train_classes[6], train_classes[7], train_classes[8], train_classes[9]])

feature_set_test = np.vstack([test_classes[0], test_classes[1], test_classes[2], test_classes[3], test_classes[4], test_classes[5], test_classes[6], test_classes[7], test_classes[8], test_classes[9]])

binary_labels_train = np.zeros((300, 10))
binary_labels_test = np.zeros((50, 10))

for i in range(300):  
    binary_labels_train[i, labels_train[i]] = 1

for i in range(50):
    binary_labels_test[i, labels_test[i]] = 1

# PLOT POINTS FOR FEATURES

plt.figure(figsize=(10,10))  
plt.scatter(feature_set_train[:,0], feature_set_train[:,1], c=labels_train, cmap='plasma', s=100, alpha=0.5)  
plt.show()


plt.figure(figsize=(10,10))  
plt.scatter(feature_set_test[:,0], feature_set_test[:,1], c=labels_test, cmap='plasma', s=100, alpha=0.5)  
plt.show()

###########		START KNN 	################

#Create KNN Classifier
no_neighbors = 5
knn = KNeighborsClassifier(n_neighbors = no_neighbors)

#Train the model using the training sets
knn.fit(feature_set_train, labels_train)

#Predict the response for test dataset
test_pred = knn.predict(feature_set_test)

# ACCURACY
# Model Accuracy, how often is the classifier correct?
print('Number of neighbors: ', no_neighbors)
print("KNN Accuracy:",metrics.accuracy_score(labels_test, test_pred))


#########################################################