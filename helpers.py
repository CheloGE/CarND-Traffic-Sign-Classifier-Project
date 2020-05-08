import numpy as np 
import cv2
import random


def get_batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return np.array(output_batches)

def rotate_images_randomly(images_batch):
    new_batch = []
    if images_batch.shape[3]==1:
        for image in images_batch:
            randNo = random.randint(1,5)
            if randNo == 1:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE).reshape(32,32,1))
            elif randNo == 2:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(32,32,1))
            elif randNo == 3:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_180).reshape(32,32,1))
            else:
                new_batch.append(image)
    elif images_batch.shape[3]==3:
        for image in images_batch:
            randNo = random.randint(1,5)
            if randNo == 1:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE).reshape(32,32,3))
            elif randNo == 2:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(32,32,3))
            elif randNo == 3:
                new_batch.append(cv2.rotate(image, cv2.ROTATE_180).reshape(32,32,3))
            else:
                new_batch.append(image)
    return np.array(new_batch)

#Image augmentation functions **********************************
def transform_image(image,ang_range,shear_range,trans_range):

    # Rotation
    Rotation = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rotation_Matrix = cv2.getRotationMatrix2D((cols/2,rows/2),Rotation,1)

    # Translation
    translation_x = trans_range*np.random.uniform()-trans_range/2
    translation_y = trans_range*np.random.uniform()-trans_range/2
    Translation_Matrix = np.float32([[1,0,translation_x],[0,1,translation_y]])

    # Shear
    points1 = np.float32([[5,5],[20,5],[5,20]])

    point1 = 5+shear_range*np.random.uniform()-shear_range/2
    point2 = 20+shear_range*np.random.uniform()-shear_range/2

    points2 = np.float32([[point1,5],[point2,point1],[5,point2]])

    shear_Matrix = cv2.getAffineTransform(points1,points2)
        
    image = cv2.warpAffine(image,Rotation_Matrix,(cols,rows))
    image = cv2.warpAffine(image,Translation_Matrix,(cols,rows))
    image = cv2.warpAffine(image,shear_Matrix,(cols,rows))
    
    return image

def gen_new_images(X_train,y_train,n_add,ang_range,shear_range,trans_range):
   
    ## checking that the inputs are the correct lengths
    assert X_train.shape[0] == len(y_train)
    # Number of classes: 43
    n_class = len(np.unique(y_train))
    X_array = []
    Y_array = []
    n_samples = np.bincount(y_train)

    for i in range(n_class):
        # Number of samples in each class

        if n_samples[i] < n_add:
            #print ("Adding %d samples for class %d" %(n_add-n_samples[i], i))
            for i_n in range(n_add - n_samples[i]):
                img_trf = transform_image(X_train[i_n],ang_range,shear_range,trans_range) 
                X_array.append(img_trf)
                Y_array.append(i)
#                print ("Number of images in class %d:%f" %(i, X_array[0])) 
           
    X_array = np.array(X_array,dtype = np.float32())
    Y_array = np.array(Y_array,dtype = np.float32())
   
    return X_array,Y_array