#! /usr/bin/env python

import rospy
from darm_msgs.msg import FingertipsPose
from darm_msgs.msg import FingersKinematicChain
from darm_msgs.msg import MDPObservation
import numpy as np

class MDPObserver:
    def __init__(self):
        self.target_pose = None
        self.kinematic_chain = None
        rospy.Subscriber("target_pose", FingertipsPose, self.target_pose_callback)
        rospy.Subscriber("darm_kinematic_chain", FingersKinematicChain, self.kinematic_chain_callback)

        self.mdp_obs_publisher = rospy.Publisher("mdp_observation", MDPObservation, queue_size=1)
        rospy.Timer(rospy.Duration(0.08), self.publish_observation, oneshot=False)

    def target_pose_callback(self, message):
            self.target_pose = message

    def kinematic_chain_callback(self, message):
            self.kinematic_chain = message

    def publish_observation(self, timerEvent):
        if self.target_pose and self.kinematic_chain:
            message = MDPObservation()
            message.digit_i = np.concatenate((self.target_pose.digit_i, 
                                             self.kinematic_chain.digit_i))
            message.digit_ii = np.concatenate((self.target_pose.digit_ii, 
                                             self.kinematic_chain.digit_ii))
            message.digit_iii = np.concatenate((self.target_pose.digit_iii, 
                                             self.kinematic_chain.digit_iii))
            message.digit_iv = np.concatenate((self.target_pose.digit_iv, 
                                             self.kinematic_chain.digit_iv))
            message.digit_v = np.concatenate((self.target_pose.digit_v, 
                                             self.kinematic_chain.digit_v))
            
            self.mdp_obs_publisher.publish(message)
        else:
            if not self.target_pose: rospy.loginfo("Current observation not published: `target_pose` message not available yet...")
            if not self.kinematic_chain: rospy.loginfo("Current observation not published: `darm_kinematic_chain` message not available yet...")

if __name__ == "__main__":
    rospy.init_node("mdp_observer")

    mdpObserver = MDPObserver()
    rospy.loginfo("Started MDP Observer node")
    rospy.spin()