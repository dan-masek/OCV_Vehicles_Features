import cv2
import numpy as np
import pickle

class KeypointData(object):
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors
        
    def save(self, file_name):
        data = []
        for keypoint, descriptor in zip(self.keypoints, self.descriptors):
            data.append([keypoint.pt
                , keypoint.size
                , keypoint.angle
                , keypoint.response
                , keypoint.octave
                , keypoint.class_id
                , descriptor])
    
        pickle.dump(data, open(file_name, "wb" ))
        
    @staticmethod
    def load(file_name):
        data = pickle.load(open(file_name, "rb" ))
        
        keypoints = []
        descriptors = []
        
        for entry in data:
            point = entry[0]
            size = entry[1]
            angle = entry[2]
            response = entry[3]
            octave = entry[4]
            class_id = entry[5]
            
            keypoints.append(cv2.KeyPoint(x=point[0],y=point[1]
                , _size=size
                , _angle=angle
                , _response=response
                , _octave=octave
                , _class_id=class_id))
            
            descriptors.append(entry[6])
            
        return KeypointData(keypoints, np.array(descriptors, np.float32))


