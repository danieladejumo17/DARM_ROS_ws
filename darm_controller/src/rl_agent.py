#! /usr/bin/env python

import os
import rospy
import numpy as np
from stable_baselines3 import SAC

from darm_msgs.msg import MDPObservation
from darm_srvs.srv import DarmAction, DarmActionRequest

class DARMRLAgent:
    def __init__(self) -> None:
        self._load_models()

        self.observation = None
        rospy.Subscriber("mdp_observation", MDPObservation, self.process_observation)

        rospy.wait_for_service("darm_stepper_action", timeout=30)
        self.action_proxy = rospy.ServiceProxy("darm_stepper_action", DarmAction)

        rospy.Timer(rospy.Duration(0.08), self.act, oneshot=False)

    def _load_models(self):
        # ==================== DIGIT I ====================
        env_tag = "di"
        run_name = f"SB3_SAC_{env_tag}_position_stiffen_2"
        run_local_dir = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_training/results/{env_tag}/{run_name}"

        model_name_i = f"{run_local_dir}/models/best/best_model"
        self.eval_model_i = SAC.load(model_name_i)

        # ==================== DIGIT II ====================
        env_tag = "dii"
        run_name = f"SB3_SAC_{env_tag}_position_stiffen_2"
        run_local_dir = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_training/results/{env_tag}/{run_name}"

        model_name_ii = f"{run_local_dir}/models/model"
        # model_name_ii = f"{run_local_dir}/models/best/best_model"
        self.eval_model_ii = SAC.load(model_name_ii)

        # ==================== DIGIT III ====================
        env_tag = "diii"
        run_name = f"SB3_SAC_{env_tag}_position_stiffen_2"
        run_local_dir = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_training/results/{env_tag}/{run_name}"

        model_name_iii = f"{run_local_dir}/models/best/best_model"
        self.eval_model_iii = SAC.load(model_name_iii)


        # ==================== DIGIT IV ====================
        env_tag = "div"
        run_name = f"SB3_SAC_{env_tag}_position_stiffen_2"
        run_local_dir = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_training/results/{env_tag}/{run_name}"

        model_name_iv = f"{run_local_dir}/models/best/best_model"
        self.eval_model_iv = SAC.load(model_name_iv)


        # ==================== DIGIT V ====================
        env_tag = "dv"
        run_name = f"SB3_SAC_{env_tag}_position_stiffen_2"
        run_local_dir = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_training/results/{env_tag}/{run_name}"

        model_name_v = f"{run_local_dir}/models/model"
        # model_name_v = f"{run_local_dir}/models/best/best_model"
        self.eval_model_v = SAC.load(model_name_v)

        rospy.loginfo("RL Models loaded successfully")

    def get_action(self, mdp_obs):
        actions = []
        actions.append(self.eval_model_i.predict(mdp_obs.digit_i, deterministic=True)[0])
        actions.append(self.eval_model_ii.predict(mdp_obs.digit_ii, deterministic=True)[0])
        actions.append(self.eval_model_iii.predict(mdp_obs.digit_iii, deterministic=True)[0])
        actions.append(self.eval_model_iv.predict(mdp_obs.digit_iv, deterministic=True)[0])
        actions.append(self.eval_model_v.predict(mdp_obs.digit_v, deterministic=True)[0])
        
        return np.concatenate(actions)
    
    def process_observation(self, message):
        self.observation = message

    def act(self, timerEvent):
        # Call a service to read the observation instead???
        if self.observation:
            action = self.get_action(self.observation)
            response = self.action_proxy(DarmActionRequest(action=action))
            # rospy.logdebug(f"Maximum step err: {np.max(response.err)}")


if __name__ == "__main__":
    rospy.init_node("rl_agent")

    darm_rl_agent = DARMRLAgent()
    rospy.loginfo("Started the DARM RL Agent Node")

    rospy.spin()