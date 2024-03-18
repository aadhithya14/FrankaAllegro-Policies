# # Main script for hand interractions 
# import cv2 
# import gym
# import numpy as np
# import os
# import torch
# import torchvision.transforms as T

# from gym import spaces
# from openteach_api import DeployAPI
# from openteach.robot.allegro.allegro_kdl import AllegroKDL
# from openteach.utils.network import ZMQCameraSubscriber
# from PIL import Image as im

# from franka_allegro.tactile_data import TactileImage, TactileRepresentation
# from franka_allegro.models import init_encoder_info
# from franka_allegro.utils import *

# class BowlUnstackingEnv(gym.Env):
#         def __init__(self, # We will use both the hand and the arm
#             tactile_out_dir,
#             tactile_model_type = 'byol',
#             host_address = "172.24.71.240",
#             camera_num = 1,
#             height = 480,
#             width = 480,
#             action_type = 'joint' # fingertip / joint
#         ):
#             # print(camera_num, "CAMERA_NUM")
#             self.width = width
#             self.height = height
#             self.view_num = camera_num

#             self.deploy_api = DeployAPI(
#                 host_address=host_address,
#                 required_data={"rgb_idxs": [camera_num], "depth_idxs": []}
#             )

#             # self.min_action, self.max_action = -5, 5 # To clip the action
#             self.home_state = dict(
#                 allegro = np.array([
#                     -0.0658244726801581, 0.11152991296986751, 0.036465840916854717, 0.29693057660614736, # Index
#                     -0.09053422635521813, 0.21657171862672447, -0.17754325611897262, 0.27011271061536507, # Middle
#                     0.012094523852233988, 0.11196786731996372, -0.017784060790178313, 0.2670852707825862, # Ring
#                     0.8499175389966154, 0.3062633015641964, 0.7989875369900138, 0.46722180902731736 # Thumb
#                 ]),
#                 kinova = np.array([-0.52456534,  0.21932657,  0.37987852,  0.04104819, -0.65619761, -0.02854434,  0.75293094])
#             )

#             self._robot = AllegroKDL()
#             # NOTE: Since the tactile observation is going to be representations it can still be
#             # considered lower than 255

#             device = torch.device('cuda:0')
#             tactile_cfg, tactile_encoder, _ = init_encoder_info(device, tactile_out_dir, 'tactile', model_type=tactile_model_type)
#             tactile_img = TactileImage(
#                 tactile_image_size = tactile_cfg.tactile_image_size, 
#                 shuffle_type = None
#             )
#             tactile_repr_dim = tactile_cfg.encoder.tactile_encoder.out_dim if tactile_model_type == 'bc' else tactile_cfg.encoder.out_dim
#             self.tactile_repr = TactileRepresentation(
#                 encoder_out_dim = tactile_repr_dim,
#                 tactile_encoder = tactile_encoder,
#                 tactile_image = tactile_img,
#                 representation_type = 'tdex'
#             )
            
#             ## WARNING: Yinlong changed
#             action_dim = 23 if action_type == 'joint' or action_type == 'robot' else 19
#             ## WARNING: Yinlong changed
#             self.action_type = action_type
#             self.action_space = spaces.Box(low = np.array([-1]*action_dim,dtype=np.float32), # Actions are 12 + 7
#                                            high = np.array([1]*action_dim,dtype=np.float32),
#                                            dtype = np.float32)
#             # self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255],dtype=np.float32), dtype = np.float32)
#             self.observation_space = spaces.Dict(dict(
#                 pixels = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255], dtype=np.float32), dtype = np.float32),
#                 tactile = spaces.Box(low = np.array([-1]*tactile_repr_dim, dtype=np.float32),
#                                      high = np.array([1]*tactile_repr_dim, dtype=np.float32),
#                                      dtype = np.float32),
#                 features = spaces.Box(low = np.array([-1]*23, dtype=np.float32),
#                                       high = np.array([1]*23, dtype=np.float32),
#                                       dtype = np.float32)
#             ))
            
#             self.image_subscriber = ZMQCameraSubscriber(
#                 host = host_address,
#                 port = 10005 + self.view_num,
#                 topic_type = 'RGB'
#             )
#             self.image_transform = T.Compose([ # No normalization just simple cropping
#                 T.Resize((480,640)),
#                 T.Lambda(self._crop_transform),
#                 T.Resize((self.height, self.width)),
#                 T.ToTensor(),
#                 T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)
#             ]) # We're not normalizing here - we normalize in the reward extraction

#             self.visualize_image_transform = T.Compose([
#                 T.Resize((480,640)),
#                 T.Lambda(self._crop_transform),
#                 T.Resize((self.height, self.width))
#             ])

#         def set_up_env(self):
#             os.environ["MASTER_ADDR"] = "localhost"
#             os.environ["MASTER_PORT"] = "29505"

#             torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
#             torch.cuda.set_device(0)

#         def _get_curr_image(self, visualize=True):
#             image, _ = self.image_subscriber.recv_rgb_image()
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = im.fromarray(image, 'RGB')
#             if visualize:
#                 img = self.visualize_image_transform(image)
#                 img = np.asarray(img)
#             else:
#                 img = self.image_transform(image)
#                 img = torch.FloatTensor(img)
#             return img # NOTE: This is for environment

#         def _crop_transform(self, image):
#             return crop_transform(image, camera_view=self.view_num)

#         def init_hand(self): # TODO: You should have dexterity_env as the base environment and then all the rest parametric
#             self.deploy_api.send_robot_action(self.home_state)

#         def step(self, action):
#             print('action.shape: {}'.format(action.shape))
#             try: 
#                 if self.action_type == 'fingertip':
#                     hand_joint_action = self._robot.get_joint_state_from_coord(
#                         action[0:3], action[3:6], action[6:9], action[9:12],
#                         self.deploy_api.get_robot_state()['allegro']['position'])
#                 else:
#                     hand_joint_action = action[:16]
                
#                 self.deploy_api.send_robot_action({
#                     'allegro': hand_joint_action, 
#                     'kinova':  action[-7:]
#                 })
#             except:
#                 print("IK error")
            
#             # Get the observations
#             obs = {}
#             features_dict = self.deploy_api.get_robot_state()
#             obs['features'] = np.concatenate(
#                 [features_dict['allegro']['position'], features_dict['kinova']],
#                 axis=0
#             )

#             obs['pixels'] = self._get_curr_image() # NOTE: Check this - you're returning non normalized things though
            
#             sensor_state = self.deploy_api.get_sensor_state()
#             tactile_values = sensor_state['xela']['sensor_values']
#             obs['tactile'] = self.tactile_repr.get(tactile_values)

#             reward, done, infos = 0, False, {'is_success': False} 

#             return obs, reward, done, infos #obs, reward, done, infos

#         def render(self, mode='rbg_array', width=0, height=0):
#             return self._get_curr_image(visualize=True)
    
#         def reset(self): 
#             self.init_hand()
#             obs = {}
#             features_dict = self.deploy_api.get_robot_state() # NOTE: having the features should be better and faster as well
#             obs['features'] = np.concatenate(
#                 [features_dict['allegro']['position'], features_dict['kinova']],
#                 axis=0
#             )
#             obs['pixels'] = self._get_curr_image()
            
#             sensor_state = self.deploy_api.get_sensor_state()
#             tactile_values = sensor_state['xela']['sensor_values']
#             obs['tactile'] = self.tactile_repr.get(tactile_values)
#             return obs

#         def get_reward():
#             pass



import numpy as np

from .env import DexterityEnv

class BowlUnstackingEnv(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            allegro = np.array([
                -0.0658244726801581, 0.11152991296986751, 0.036465840916854717, 0.29693057660614736, # Index
                -0.09053422635521813, 0.21657171862672447, -0.17754325611897262, 0.27011271061536507, # Middle
                0.012094523852233988, 0.11196786731996372, -0.017784060790178313, 0.2670852707825862, # Ring
                0.8499175389966154, 0.3062633015641964, 0.7989875369900138, 0.46722180902731736 # Thumb
            ]),
            kinova = np.array([-0.52456534,  0.21932657,  0.37987852,  0.04104819, -0.65619761, -0.02854434,  0.75293094])
        )
