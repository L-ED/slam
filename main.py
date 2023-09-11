import argparse
import numpy as np
import os, cv2
import numpy as np
import threading
import queue
import cv2

import OpenGL.GL as gl
import pangolin

from time import sleep


class PangolinPlotter:
    MSG_EXIT = "exit"
    MSG_NEW_POINTS = "new point"
    MSG_NEW_POSE = "new pose"
    MSG_NEW_GT = "new gt"

    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, args=(self.q,), kwargs={})

    def start(self):
        assert not self.thread.is_alive()
        self.thread.start()

    def finish(self):
        assert self.thread.is_alive()
        self.q.put((self.MSG_EXIT, ))
        self.thread.join()

    def draw_3d_pose(self, pose_mtx):
        assert pose_mtx.shape == (4, 4)
        self.q.put((self.MSG_NEW_POSE, pose_mtx))

    def draw_gt_pose(self, pose_mtx):
        assert pose_mtx.shape == (4, 4)
        self.q.put((self.MSG_NEW_GT, pose_mtx))

    def draw_3d_points(self, point_array, color_array=None):
        assert len(point_array.shape) == 2 and point_array.shape[1] == 3
        assert color_array is None or len(color_array) == len(point_array)

        # Try and transform the coordinate system
        # pangolin_point_array = point_array.copy()
        # pangolin_point_array[:, 0], pangolin_point_array[:, 1], pangolin_point_array[:, 2] = point_array[:, 1], point_array[:, 2], point_array[:, 0]

        self.q.put((self.MSG_NEW_POINTS, point_array, color_array))

    @staticmethod
    def _follow_camera(scam, pose):
        scam.Follow(pangolin.OpenGlMatrix(pose), True)

    @staticmethod
    def _draw_cameras(camera_poses, color=(0.0, 0.0, 1.0)):
        if camera_poses:
            gl.glLineWidth(1)
            # gl.glColor3f(0.0, 0.0, 1.0)
            gl.glColor3f(*color)
            pangolin.DrawCameras(np.array(camera_poses))

    @staticmethod
    def _draw_points(list_of_arrays, list_of_color_arrays):
        gl.glPointSize(2)
        default_color = (1., 0., 0,)
        for point_array, color_array in zip(list_of_arrays, list_of_color_arrays):
            if color_array is not None:
                pangolin.DrawPoints(point_array, color_array)
            else:
                gl.glColor3f(*default_color)
                pangolin.DrawPoints(point_array)

    def _run(self, q):
        camera_poses = []
        camera_gt = []
        point_arrays = []
        color_arrays = []

        pangolin.CreateWindowAndBind("3D Plot", 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            # From http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html#ae25baa30091d20a3119459811f8d5b35:
            # OpenGlMatrixSpec pangolin::ProjectionMatrix
            # int w, int h,
            # double fv, fu - focal point
            # double u0, v0 - principal point offset (also known as cx, cy)
            # double zNear, double zFar - distance to near and far clipping planes. Objects close and farther are invisible to the camera
            pangolin.ProjectionMatrix(1024, 768, 420, 420, 1024 // 2, 768 // 2, 0.2, 10000.0),

            # Params below:
            # ex, ey, ez - rotates view - eyeX, eyeY, eyeZ?
            # lx, ly, lz - translates view - look vector?
            # ux, uy, uz - direction of up axis?
            # Yiss - I think I figured it out. The observation vector (idk the actual term) simply points from (ex,ey,ez) to (lx,ly,lz)
            pangolin.ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin.AxisDirection.AxisNegY)
        )
        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, 1024. / 768.)
        dcam.SetHandler(handler)
        # dcam.Resize(pangolin.Viewport(0, 0, 1024 * 2, 768 * 2))  # From GeoHotz TwitchSLAM, this fixed his intermittent
        # display issues somehow, though he doensn't know why.
        # dcam.Activate(scam)

        while not pangolin.ShouldQuit():
            # Process messages
            try:
                msg = q.get_nowait()
                if msg[0] == self.MSG_EXIT:
                    break
                elif msg[0] == self.MSG_NEW_POINTS:
                    point_array, color_array = msg[1], msg[2]
                    point_arrays.append(point_array)
                    color_arrays.append(color_array)
                    continue
                elif msg[0] == self.MSG_NEW_POSE:
                    camera_pose = msg[1]
                    # self._follow_camera(scam, camera_pose)
                    camera_poses.append(camera_pose)
                    continue
                elif msg[0] == self.MSG_NEW_GT:
                    camera_pose = msg[1]
                    # self._follow_camera(scam, camera_pose)
                    camera_gt.append(camera_pose)
                    continue
            except queue.Empty:
                pass

            # Clear view
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            dcam.Activate(scam)

            self._draw_points(point_arrays, color_arrays)
            self._draw_cameras(camera_poses)
            self._draw_cameras(camera_gt, color=(1.0, 0.0, 0.0))

            pangolin.FinishFrame()
            sleep(0.1)


class Stream_Capture:
    def __init__(self, visual_stream, visual_type):
        if visual_type == 'video':
            NotImplementedError("Not read opyins for video stream")
        elif visual_type == 'photo':
            self.shots = sorted(os.listdir(visual_stream))
            self.dir = visual_stream

    def __call__(self):

        for imgpath in self.shots:
            fullpath = os.path.join(self.dir, imgpath)
            img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
            yield img

        
class Point:
    def __init__(self):
        self.img_idx_map = {}
        self.coords = None
    
    def add_idx(self, img_idx, keypoint_idx):
        self.keypoints[img_idx]=keypoint_idx

    def add_coord(self, coord):
        self.coords = coord

    


class Monocular_VSLAM:
    def __init__(self, visual_stream, visual_type, camera_matrix=None, ground_truth=None):

        if ground_truth is not None:
            with open(ground_truth, 'r') as gt:
                self.ground_truth = gt.readlines()
            
        self.stream = Stream_Capture(visual_stream, visual_type)
        self.plotter = PangolinPlotter()
        self.plotter.start()
        self.scale = np.array(1.0)
        self.transformations = np.array([])
        self.base = np.array([0, 0, 0])
        self.intrinsic = camera_matrix
        if camera_matrix is None:
            self.intrinsic = np.array(
                [
                    [7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0],
                    [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0],
                    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0]
                ]
            )
        # self.map = np.array([])
        self.descriptors = []
        self.keypoints = []
        self.matches = {}
        self.poses = [np.eye(4)]
        self.pose = np.eye(4)
        self.scale=1

        self.projections = [self.intrinsic]
        self.Feature_Detector = cv2.SIFT_create()

        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        self.Feature_Matcher = cv2.FlannBasedMatcher(
            index_params,search_params)
        # self.Feature_Matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    def run(self):

        for i, img in enumerate(self.stream()):
            keypoints, descriptors = self.take_descriptors(img)
            if i > 0:
                matches = self.find_matches(
                    descriptors, self.descriptors[-1]
                )

                T, cur_pts, prev_pts = self.get_transformation(
                    matches, keypoints, self.keypoints[-1]
                )

                # new_pose = np.dot(self.poses[-1], T)
                new_pose = np.dot(self.pose, T)
                self.pose = new_pose

                new_points, points_filter = self.triangulate_points(
                    new_pose, cur_pts, prev_pts
                )


                if i==2:
                    front2d_pts = cur_pts[points_filter]
                    self.save_points(front2d_pts, new_points)

                # self.update_map(new_points)
                if i==3:
                    
                    self.update_scale(
                        prev_pts[points_filter], 
                        new_points
                    )
                    self.update_transformations()

                self.update_map(
                    img, new_pose, new_points, cur_pts, points_filter
                )
                self.poses.append(new_pose)

            if self.ground_truth is not None:
                gt = np.eye(4)
                pose = np.array([
                    float(el) for el in 
                    self.ground_truth[i].strip().split(' ')
                ]).reshape((3, 4))
                gt[:3 , :] = pose
                self.plotter.draw_gt_pose(gt)

            
            self.descriptors.append(descriptors)
            self.keypoints.append(keypoints)
            # self.poses.append(new_pose)

            # self.detections.append(detections)

    def save_points(self, pts2d, pts3d):
        self.saved_points = {
            tuple(coord2d):coord3d
            for coord2d,coord3d in zip(pts2d, pts3d)
        }

    
    def update_scale(self, prev2dpts, new3dpoints):
        pts_corr = []
        for prev2dpt, new3dpt in zip(
            prev2dpts, new3dpoints):

            pt_key = tuple(prev2dpt)

            if pt_key in self.saved_points:
                pts_corr.append(
                    [
                        self.saved_points[pt_key],
                        new3dpt
                    ]
                )

        pts_corr = np.array(pts_corr)

        indexes = np.linspace(
            0, len(pts_corr)-1, len(pts_corr), dtype=np.int32)
        # for iteration in range(5):

        obs_pts_id = np.random.choice(
            indexes,  
            size=len(pts_corr)*0.4)

        ref = pts_corr[
            obs_pts_id[:len(obs_pts_id)//2]
        ]

        intrst = pts_corr[
            obs_pts_id[len(obs_pts_id)//2:]
        ]

        assert len(ref) == len(intrst)

        dists = np.sum((intrst - ref)**2, axis=-1)**0.5

        self.scale = np.mean(dists[:, 0]/dists[:, 1])


    def update_tranformations(self):
            
        for T in self.poses:
            T[:3, 3:]/=self.scale 
            self.pose
            





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
            if m.distance < 0.3*n.distance:
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
        # cur_ids = [matches[:].queryIdx]
        # prev_ids = [matches[:].trainIdx]

        assert len(matches)

        cur_points, prev_points = [],[]
        for match in matches:
            cur_points.append(
                cur_kp[match.queryIdx].pt
            )
            prev_points.append(
                prev_kp[match.trainIdx].pt
            )

        assert len(prev_points)==len(cur_points) and len(prev_points)>0

        cur_points=np.array(cur_points, dtype=np.int32)
        prev_points=np.array(prev_points, dtype=np.int32)

        E, mask = cv2.findEssentialMat(
            cur_points, 
            prev_points,
            self.intrinsic[:, :3]
        )
        
        inlier_indices = mask[:, 0] == 1
        
        cur_points = cur_points[inlier_indices]
        prev_points = prev_points[inlier_indices]

        retval, R, t, _ = cv2.recoverPose(
            E, cur_points, prev_points
        )

        if t[2]<0: # wrong translation fix
            t[2]*=-1
        
        cam_R = R.T
        cam_t = -np.dot(cam_R,t)
        
        T = np.eye(4)

        T[:3, :3] = R
        T[:3, 3:] = t

        return T, cur_points, prev_points
    

    def triangulate_points(self, new_pose, cur_points, prev_points):

        new_proj_matrix = np.dot(self.intrinsic, new_pose)

        points4d = cv2.triangulatePoints(
            new_proj_matrix, 
            self.projections[-1], 
            cur_points.T.astype(np.float32), 
            prev_points.T.astype(np.float32)
        ).T

        self.projections.append(new_proj_matrix)

        point_filter = points4d[:, 3] < 0
        points4d = points4d[point_filter]
        return points4d[:, :3] / points4d[:, 3:], point_filter
    

    def update_map(self, img, pose, points3d, pts2d, point_filter):

        pts_3d_colors = np.array([np.full(3, img[y][x]) for x, y in pts2d]) / 255.
        pts_3d_colors = pts_3d_colors[point_filter]
        pts_3d_colors[:, 0], pts_3d_colors[:, 2] = pts_3d_colors[:, 2], pts_3d_colors[:, 0].copy()

        self.plotter.draw_3d_pose(pose)
            # plotter_3d.draw_3d_pose(pangolin_pose_mtx)
        # self.plotter.draw_3d_points(points3d, pts_3d_colors)








if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visual_stream')
    parser.add_argument('-t', '--visual_stream_type')
    parser.add_argument('-o', '--odometry_stream', default=None)
    parser.add_argument('-ot', '--odometry_type', default=None)

    args = parser.parse_args()

    slam_system = Monocular_VSLAM(
        '/home/led/robotics/computer_vision/datasets/KITTI/data_odometry_gray/dataset/sequences/02/image_0',
        'photo', 
        ground_truth='/home/led/robotics/computer_vision/datasets/KITTI/data_odometry_poses/dataset/poses/02.txt'
    )

    slam_system.run()
    # system(
    #     visual_stream=args.visual_stream,
    #     visual_type=args.visual_stream_type,
    #     # odometry_stream=args.odometry_stream,
    #     # odometry_type=args.odometry_type
    # )
