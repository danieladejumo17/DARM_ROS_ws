#! /usr/bin/env python

import rospy
from darm_msgs.msg import FingertipsPose
from darm_srvs.srv import DarmJointState, DarmJointStateRequest
from darm_srvs.srv import DarmSimGoToJointState, DarmSimGoToJointStateRequest
from darm_srvs.srv import UpdateTargetViz, UpdateTargetVizRequest
from std_srvs.srv import Empty, EmptyResponse
from darm_gym_env import DARMEnv
import numpy as np


class SimpleGoalPublisher():
    def __init__(self) -> None:
        self.pose_gen_darm_robot = DARMEnv(render_mode=None, 
                                           action_time=0.08, 
                                           hand_name="hand1", 
                                           digits=["i", "ii", "iii", "iv", "v"], 
                                           ignore_load_start_states=True)

        self.publisher = rospy.Publisher("target_pose", FingertipsPose, queue_size=1)
        
        # Wait for darm_sim services - darm_joint_state, goto_joint_state, update_target_viz
        rospy.loginfo("Waiting for service `darm_joint_state`")
        rospy.wait_for_service("darm_joint_state", timeout=5)

        rospy.loginfo("Waiting for service `goto_joint_state`")
        rospy.wait_for_service("goto_joint_state", timeout=5)

        rospy.loginfo("Waiting for service `update_target_viz`")
        rospy.wait_for_service("update_target_viz", timeout=5)

        # Service Proxies
        self.darm_joint_state_proxy = rospy.ServiceProxy("darm_joint_state", DarmJointState)
        self.goto_joint_state_proxy = rospy.ServiceProxy("goto_joint_state", DarmSimGoToJointState)
        self.update_goal_viz_proxy = rospy.ServiceProxy("update_target_viz", UpdateTargetViz)

        # Initialize the goal message
        self.new_goal()

        # New Goal Service
        rospy.Service("simple_new_goal", Empty, self.new_goal_callback)
        rospy.loginfo("Started Service `simple_new_goal`")

        # Timer to publish goal at specified rate
        def publish_goal(timerEvent):
            self.publish()
        rospy.Timer(rospy.Duration(0.08), publish_goal, oneshot=False)

    def new_goal(self):
        """Generate a new goal which we published on next publish calls"""
        # Get the current robot joint state
        rospy.loginfo("Generating new goal")
        response = self.darm_joint_state_proxy(DarmJointStateRequest())
        self.pose_gen_darm_robot.forward(response.joint_state)

        result = self.pose_gen_darm_robot.generate_target(n_trials=200)
        target_pose = None
        if result:
            rospy.loginfo("Valid goal created from current robot pose...")
            _, _, target_pose = result
        else:
            rospy.loginfo("Calling for new robot pose to create valid goal...")
            _, joint_state, target_pose = self.pose_gen_darm_robot.generate_start_state()

            rospy.loginfo("Calling service `goto_joint_state`...")
            self.goto_joint_state_proxy(DarmSimGoToJointStateRequest(joint_state=joint_state))
        
        self.goal = FingertipsPose(digit_i=target_pose[0],
                                        digit_ii=target_pose[1],
                                        digit_iii=target_pose[2],
                                        digit_iv=target_pose[3],
                                        digit_v=target_pose[4])
        
        self.update_goal_viz_proxy(UpdateTargetVizRequest(target=self.goal))
        rospy.loginfo("New goal generated")
        
    def publish(self, goal = None):
        self.publisher.publish(goal or self.goal)

    def new_goal_callback(self, request):
        self.new_goal()
        return EmptyResponse()

    def __del__(self):
        print("Closing DARM Simulation")
        self.pose_gen_darm_robot.close()

if __name__ == "__main__":
    rospy.init_node("simple_goal_publisher", log_level=rospy.INFO)

    rospy.loginfo("Starting Simple Goal Publisher Node")
    goal_publisher = SimpleGoalPublisher()
    rospy.loginfo("Started Simple Goal Publisher Node")

    rospy.spin()
