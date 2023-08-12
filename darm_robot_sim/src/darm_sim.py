#! /usr/bin/env python

import rospy
from darm_gym_env import DARMEnv
from darm_msgs.msg import FingersKinematicChain
from darm_msgs.msg import FingertipsPose
from darm_srvs.srv import DarmJointState, DarmJointStateResponse
from darm_srvs.srv import DarmSimGoToJointState, DarmSimGoToJointStateResponse
from darm_srvs.srv import DarmAction, DarmActionResponse
from darm_srvs.srv import UpdateTargetViz, UpdateTargetVizResponse
import numpy as np
import time

class DARMSim:
    def __init__(self) -> None:
        self.darm_robot = DARMEnv(render_mode="human",
                                  action_time=0.08,
                                  hand_name="hand1",
                                  digits=["i", "ii", "iii", "iv", "v"],
                                  ignore_load_start_states=True)
        
        # Topic Publishers
        self.darm_kinematic_chain_publisher = rospy.Publisher("darm_kinematic_chain", 
                                                              FingersKinematicChain, 
                                                              queue_size=1)
        self.darm_pose_publisher = rospy.Publisher("darm_pose", FingertipsPose, queue_size=1)
        
        # darm_joint_state service
        rospy.Service("darm_joint_state", DarmJointState, self.joint_state_srv_callback)
        rospy.loginfo("Started Service `darm_joint_state`")

        # goto_joint_state service
        rospy.Service("goto_joint_state", DarmSimGoToJointState, self.goto_joint_state)
        rospy.loginfo("Started Service `goto_joint_state`")

        # Service to update target visualization
        rospy.Service("update_target_viz", UpdateTargetViz, self.update_target_viz)
        rospy.loginfo("Started Service `update_target_viz`")

        # DARM Action Service
        rospy.Service("darm_stepper_action", DarmAction, self.action_callback)
        rospy.loginfo("Started Service `darm_stepper_action`")

        # Timers to render, and publish topics
        # rospy.Timer(rospy.Duration(0.05), self.render, oneshot=False)
        rospy.Timer(rospy.Duration(0.08), self.publish_darm_kinematic_chain, oneshot=False)
        rospy.Timer(rospy.Duration(0.08), self.publish_darm_pose, oneshot=False)
    
    def publish_darm_kinematic_chain(self, timerEvent):
        message = FingersKinematicChain(digit_i = self.darm_robot.get_finger_frames_pos("i"),
                                        digit_ii = self.darm_robot.get_finger_frames_pos("ii"),
                                        digit_iii = self.darm_robot.get_finger_frames_pos("iii"),
                                        digit_iv = self.darm_robot.get_finger_frames_pos("iv"),
                                        digit_v = self.darm_robot.get_finger_frames_pos("v"),)
        self.darm_kinematic_chain_publisher.publish(message)

    def publish_darm_pose(self, timerEvent):
        message = FingertipsPose(digit_i=self.darm_robot.get_fingertip_pose("i"),
                                 digit_ii=self.darm_robot.get_fingertip_pose("ii"),
                                 digit_iii=self.darm_robot.get_fingertip_pose("iii"),
                                 digit_iv=self.darm_robot.get_fingertip_pose("iv"),
                                 digit_v=self.darm_robot.get_fingertip_pose("v"),)
        self.darm_pose_publisher.publish(message)

    def render(self, timerEvent):
        self.darm_robot.render()

    def joint_state_srv_callback(self, request):
        return DarmJointStateResponse(joint_state=self.darm_robot.joint_state())
    
    def update_target_viz(self, request):
        print("Updating target viz...")
        print(request)
        message = request.target
        target_pose = np.array([message.digit_i,
                                message.digit_ii,
                                message.digit_iii,
                                message.digit_iv,
                                message.digit_v,])
        
        # rospy.logdebug("Updating target pose viz...")
        self.darm_robot.set_fingertips_mocap_pose(target_pose)
        
        return UpdateTargetVizResponse(success=True)

    def goto_joint_state(self, request):
        # Reset first
        self.darm_robot.reset_mujoco_model(mj_forward=False)

        # Go to Joint State
        self.darm_robot.forward(request.joint_state, mj_forward=False)
        return DarmSimGoToJointStateResponse(success=True)

    def action_callback(self, message):
        err = self.darm_robot.act(np.array(message.action))
        return DarmActionResponse(err=err)

    def __del__(self):
        print("Closing DARM Simulation")
        self.darm_robot.close()

if __name__ == "__main__":
    rospy.init_node("darm_robot_sim")

    darm_sim = DARMSim()
    rospy.loginfo("Started DARM Robot Simulation")
    # rospy.spin()

    rate = rospy.Rate(25)
    while not rospy.is_shutdown():
        darm_sim.render(None)  # if there is a way to do this asynchronously
        rate.sleep()
