


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
    def _draw_cameras(camera_poses):
        if camera_poses:
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
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
                    self._follow_camera(scam, camera_pose)
                    camera_poses.append(camera_pose)
                    continue
            except queue.Empty:
                pass

            # Clear view
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            dcam.Activate(scam)

            self._draw_points(point_arrays, color_arrays)
            self._draw_cameras(camera_poses)

            pangolin.FinishFrame()
            sleep(0.1)
