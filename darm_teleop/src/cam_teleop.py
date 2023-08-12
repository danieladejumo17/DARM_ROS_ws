#! /usr/bin/env python3

import sys
sys.path.append('/home/daniel/DARM/darm_ws/src/darm_teleop/src/minimal_hand_v2')

import rospy
from darm_msgs.msg import FingertipsPose, FingertipsWristPose
from darm_srvs.srv import UpdateTargetViz, UpdateTargetVizRequest

import time

import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat

from minimal_hand_v2 import config
from minimal_hand_v2.capture import OpenCVCapture
from minimal_hand_v2.hand_mesh import HandMesh
from minimal_hand_v2.kinematics import mpii_to_mano, MPIIHandJoints
from minimal_hand_v2.utils import OneEuroFilter, imresize
from minimal_hand_v2.wrappers import ModelPipeline
from minimal_hand_v2.utils import *


class CamTeleop:
    def __init__(self, capture, window_size=700, distance_scale=100):
        """
        Launch an application that reads from a webcam and estimates hand pose at
        real-time.

        The capture fingertips poses are published to the `target_pose` topic

        The captured hand must be the right hand, but will be flipped internally
        and rendered.

        Parameters
        ----------
        capture : object
            An object from `capture.py` to read capture stream from.
        """

        self.IK_UNIT_LENGTH = config.IK_UNIT_LENGTH
        self.scale = self.IK_UNIT_LENGTH*distance_scale # dm to m, then m to cm

        self.capture = capture
        self.window_size = window_size

        self.setup_output_viz()
        self.setup_input_viz()

        ############ misc ############
        self.mesh_smoother = OneEuroFilter(4.0, 0.0)
        self.clock = pygame.time.Clock()
        self.model = ModelPipeline()

        self.target_position = None # [21, 3]
        self.target_orientation = None # [21, 4]
        
        self.prev_time = time.time()
        self.process_frame()

        # Update Target Viz Service
        rospy.loginfo("Waiting for service `update_target_viz`") 
        rospy.wait_for_service("update_target_viz", timeout=5) 
        rospy.loginfo("`update_target_viz` available") 
        self.update_target_viz_proxy = rospy.ServiceProxy("update_target_viz", UpdateTargetViz) 

        # Target Pose Publisher
        self.target_pose_publisher = rospy.Publisher("target_pose", FingertipsPose, queue_size=1)
        # rospy.Timer(rospy.Duration(0.08), self.publish_target_pose, oneshot=False)
        rospy.Timer(rospy.Duration(5), self.publish_target_pose, oneshot=False)
        
        while True:
            self.process_frame()

            if rospy.is_shutdown():
                break

    def transform_position_xyz_mpii(self, xyz_mpii):
        """
        - Transforms the distance to be in the frame of the wrist (RC Joint)
        - Convert the distance reading to cm
        - xyz_mpii : [X,Z,Y]. +x is towards the ulna side, +Z is downwards, +y is towards the posterior side
        """
        
        res = (xyz_mpii - xyz_mpii[0])*self.scale*[-1, -1, 1]

        z = res[:, 1].copy()
        y = res[:, 2].copy()
        res[:,1] = y
        res[:, 2] = z

        return res
    
    def transform_orientation_obs(self, obs):
        # obs [[X, Z, Y, W],...]

        w = obs[:, 3].copy()
        xzy = obs[:, 0:3].copy()

        obs[:, 3] = np.cos(w/2)
        obs[:, 0:3] = xzy*np.reshape(np.sin(w/2), (-1, 1))

        # IN AS (w, x, z, y)
        # res = obs*[1, -1, -1, 1] # (w, x, z, y)

        # IN AS (x, z, y, w)
        res = obs*[-1, -1, 1, 1] # (x, z, y, w)
        res = np.roll(res, shift=1, axis=-1) # (w, x, z, y)

        # ============
        # OUT AS (x, y, z, w)
        # res = np.roll(res, shift=-1, axis=-1) # (x, z, y, w)
        # z = res[:, 1].copy()
        # y = res[:, 2].copy()
        # res[:,1] = y
        # res[:, 2] = z

        # OUT AS (w, x, y, z)
        # res (w, x, z, y)  
        z = res[:, 2].copy()
        y = res[:, 3].copy()
        res[:,2] = y
        res[:, 3] = z

        return res

    def setup_output_viz(self):
        ############ output visualization ############
        self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems

        self.hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(self.hand_mesh.faces)
        self.mesh.vertices = \
            o3d.utility.Vector3dVector(np.matmul(self.view_mat, self.hand_mesh.verts.T).T * 1000)
        self.mesh.compute_vertex_normals()

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(
            width=self.window_size + 1, height=self.window_size + 1,
            window_name='Minimal Hand - output'
        )
        self.viewer.add_geometry(self.mesh)

        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic
        cam_params.intrinsic.set_intrinsics(
            self.window_size + 1, self.window_size + 1, config.CAM_FX, config.CAM_FY,
            self.window_size // 2, self.window_size // 2
        )
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json('./minimal_hand_v2/render_option.json')
        self.viewer.update_renderer()

    def setup_input_viz(self):
        ############ input visualization ############
        pygame.init()
        self.display = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Minimal Hand - input')

    def process_frame(self):
        frame_large = self.capture.read()

        if frame_large is None:
            return
        
        if frame_large.shape[0] > frame_large.shape[1]:
            margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
            frame_large = frame_large[margin:-margin]
        else:
            margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
            frame_large = frame_large[:, margin:-margin]

        frame_large = np.flip(frame_large, axis=1).copy()
        frame = imresize(frame_large, (128, 128))

        xyz_mpii, theta_mpii = self.model.process(frame)
        self.target_position = xyz_mpii
        self.target_orientation = theta_mpii

        # def analyze():
        #     if time.time() - self.prev_time > 0.5:
                
        #         np.set_printoptions(precision=2, suppress=True, floatmode="maxprec_equal", )

        #         # print(self.transform_position_xyz_mpii(xyz_mpii)[[8, 12, 16, 20]])
        #         print(theta_mpii[[8]]) # , 12, 16, 20

        #         self.prev_time = time.time()

        # analyze()


        theta_mano = mpii_to_mano(theta_mpii)

        v = self.hand_mesh.set_abs_quat(theta_mano)
        v *= 2  # for better visualization
        v = v * 1000 + np.array([0, 0, 400])
        v = self.mesh_smoother.process(v)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.hand_mesh.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, v.T).T)
        self.mesh.paint_uniform_color(config.HAND_COLOR)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()
        self.viewer.update_geometry(self.mesh)

        self.viewer.poll_events()

        self.display.blit(
            pygame.surfarray.make_surface(
                np.transpose(
                    imresize(frame_large, (self.window_size, self.window_size)
                                ), (1, 0, 2))
            ),
            (0, 0)
        )
        pygame.display.update()

        # if keyboard.is_pressed("esc"):
        #   break

        self.clock.tick(20)

    def publish_target_pose(self, timerEvent):
        # np.set_printoptions(precision=2, suppress=True, floatmode="maxprec_equal", )
        # print("Orientation:")
        # print(self.target_orientation)


        trf_target_position = self.transform_position_xyz_mpii(self.target_position)

        def angles_from_theta(child_pos, parent_pos, y_rot=False):
            delta = trf_target_position[child_pos] - trf_target_position[parent_pos]
            if delta[2] <= 0:
                thX = 90
            else:
                thX = np.degrees(np.arctan(delta[1]/delta[2]))
            if not y_rot: return thX

            thY = np.degrees(np.arctan(delta[0]/delta[2]))
            return thX, thY

        palm_theta = angles_from_theta(9, 0, y_rot=True)
        pp_iii_theta = angles_from_theta(10, 9, y_rot=True)

        print(f"Palm Rot: X:{palm_theta[0]} Y:{palm_theta[1]}")
        print(f"PP III Rot: X:{pp_iii_theta[0]} Y:{pp_iii_theta[1]}")

        trf_target_orientation = self.transform_orientation_obs(self.target_orientation)

        message = FingertipsPose(
            digit_i=np.concatenate((trf_target_position[4], trf_target_orientation[4])),
            digit_ii=np.concatenate((trf_target_position[8], trf_target_orientation[8])),
            digit_iii=np.concatenate((trf_target_position[12], trf_target_orientation[12])),
            digit_iv=np.concatenate((trf_target_position[16], trf_target_orientation[16])),
            digit_v=np.concatenate((trf_target_position[20], trf_target_orientation[20]))
            )
        
        self.update_target_viz_proxy(UpdateTargetVizRequest(target=message))
        self.target_pose_publisher.publish(message)

if __name__ == '__main__':
    rospy.init_node("cam_teleop", log_level=rospy.INFO)
    cam_teleop = CamTeleop(OpenCVCapture())