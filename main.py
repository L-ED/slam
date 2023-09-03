import argparse
import numpy as np
import os, cv2

class Stream_Capture:
    def __init__(self, visual_stream, visual_type) -> None:
        if visual_type == 'video':
            NotImplementedError("Not read opyins for video stream")
        elif visual_type == 'photo':
            self.shots = os.listdir(visual_stream)

    def __call__(self):

        for imgpath in self.shots:
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            yield img

        
class Point:
    def __init__(self) -> None:
        self.img_idx_map = {}
        self.coords = None
    
    def add_idx(self, img_idx, keypoint_idx):
        self.keypoints[img_idx]=keypoint_idx

    def add_coord(self, coord):
        self.coords = coord

    


class Monocular_VSLAM:
    def __init__(self, visual_stream, visual_type, camera_matrix=None):
        self.stream = Stream_Capture(visual_stream, visual_type)
        self.scale = np.array(1.0)
        self.transformations = np.array([])
        self.base = np.array([0, 0, 0])
        self.camera_matrix = camera_matrix
        # self.map = np.array([])
        self.descriptors = []
        self.keypoints = []
        self.matches = {}
        self.Feature_Detector = cv2.ORB_create()

        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        self.Feature_Matcher = cv2.FlannBasedMatcher(
            index_params,search_params)


    def run(self):

        for i, img in enumerate(self.stream()):
            keypoints, descriptors = self.take_descriptors(img)
            if i > 0:
                matches = self.find_matches(
                    descriptors, self.descriptors[-1]
                )

                T = self.get_transformation(
                    matches, keypoints, self.keypoints[-1]
                )
                new_points = self.triangulate_points(
                    T, matches
                )

                # self.update_map(new_points)
                if i==2:
                    self.update_scale(matches)
                    self.update_transformations()

                self.update_map(new_points)
            
            self.detections.append(detections)


    def take_descriptors(self, image):
        return self.Feature_Detector.detectAndCompute(
            image,None)
    

    def find_matches(self, cur_features, prev_features):
        matches = self.Feature_Matcher.knnMatch(
            cur_features,
            prev_features,
            k=2
        )
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
    
        return np.asarray(good)


    def get_transformation(self, matches, cur_kp, prev_kp):
        '''
        https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution
        https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
        https://ia601408.us.archive.org/view_archive.php?archive=/7/items/DIKU-3DCV2/DIKU-3DCV2.zip&file=DIKU-3DCV2%2FHandouts%2FLecture16.pdf
        Create transformation matrix from matches between shots
        Find essential matrix from points and kamera matrix
        Decompose Essential matrix using SVD, E = R[t]x, where [t]x - representation of cross product
        Also E = U*diag(1, 1, 0)*V.T = U*([0,0,-1]x*Y)*V.T = (U*[0,0,-1]x*U.T)*(U*Y*V.T)
        Real R and T https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/?answer=183752#post-id-183752
        

            https://nghiaho.com/?p=2379#motion
            https://gitlab.com/LinasKo/visual-slam-practice/-/blob/master/src/main.py?ref_type=heads
        '''
        cur_ids = [matches[:].queryIdx]
        prev_ids = [matches[:].trainIdx]

        cur_points = np.array([
            cur_kp[i].pt for i in cur_ids
        ])
        
        prev_points = np.array([
            prev_kp[i].pt for i in prev_ids
        ])

        E, mask = cv2.findEssentialMat(
            cur_points, prev_points)
        
        inlier_indices = mask[:, 0] == 1
        
        cur_points = cur_points[inlier_indices]
        prev_points = prev_points[inlier_indices]

        retval, r_vec, t_vec, _ = cv2.recoverPose(
            E, cur_points, prev_points)
        









if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visual_stream')
    parser.add_argument('-t', '--visual_stream_type')
    parser.add_argument('-o', '--odometry_stream', default=None)
    parser.add_argument('-o', '--odometry_type', default=None)

    args = parser.parse_args()

    # system(
    #     visual_stream=args.visual_stream,
    #     visual_type=args.visual_stream_type,
    #     # odometry_stream=args.odometry_stream,
    #     # odometry_type=args.odometry_type
    # )
