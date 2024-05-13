# Script to use some of the deployment wrappers and apply the actions
import hydra
import torch 
import sys

from openteach_api.api import DeployAPI # This import could be changed depending on how it's used
from openteach.utils.timer import FrequencyTimer
from omegaconf import DictConfig

class Deploy:
    def __init__(self, cfg, deployed_module):
        self.module = deployed_module
        required_data = {
            'rgb_idxs': [0],
            'depth_idxs': [0]
        }
        self.deploy_api = DeployAPI(
            host_address = '172.24.71.206',
            required_data = required_data
        )
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.data_reprs = cfg.data_representations
        self.device = torch.device('cuda:0')
        self.frequency_timer = FrequencyTimer(cfg.frequency)

    def solve(self):
        sys.stdin = open(0) # To get inputs while spawning multiple processes

        while True:
            
            try:

                if self.cfg['loop']:
                    self.frequency_timer.start_loop()

                print('\n***************************************************************')
                print('\nGetting state information...') 

                # Get the robot state and the tactile info
                if 'allegro' in self.data_reprs or 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
                    robot_state = self.deploy_api.get_robot_state()
                if 'tactile' in self.data_reprs:
                    sensor_state = self.deploy_api.get_sensor_state()

                state_dict = dict()
                for data in self.data_reprs:
                    if data == 'allegro':
                        state_dict['allegro'] = robot_state['allegro']['position']
                        state_dict['torque'] = robot_state['allegro']['effort']
                    if data == 'franka' or data == 'kinova':
                        state_dict[data] = robot_state[data]
                    if data == 'tactile': 
                        if 'palm_sensor_values' in sensor_state['xela'] and 'fingertip_sensor_values' in sensor_state['xela'] and 'finger_sensor_values' in sensor_state['xela']: # It is the curved fingers
                            state_dict[data] = dict(
                                finger_values = sensor_state['xela']['finger_sensor_values'],
                                fingertip_values = sensor_state['xela']['fingertip_sensor_values'],
                                palm_values = sensor_state['xela']['palm_sensor_values']
                            )
                        elif 'sensor_values' in sensor_state['xela']: # It is regular fingers
                            state_dict[data] = sensor_state['xela']['sensor_values']

                pred_action = self.module.get_action(
                    state_dict=state_dict,
                    visualize=self.cfg['visualize']
                )
                if not self.cfg['loop']:
                    register = input('\nPress a key to perform the action...')

                action_dict = dict() 
                for robot in pred_action.keys():
                    action_dict[robot] = pred_action[robot]
                self.deploy_api.send_robot_action(action_dict)

                if self.cfg['loop']: 
                    self.frequency_timer.end_loop()

            except:
                self.module.save_deployment() # This is supposed to save all the representaitons and run things 
                break

@hydra.main(version_base=None, config_path='franka_allegro/configs', config_name='deploy')
def main(cfg : DictConfig) -> None:

    deployer = hydra.utils.instantiate(
        cfg.deployer,
        data_path = cfg.data_path,
        deployment_dump_dir = cfg.deployment_dump_dir
    )
    deploy = Deploy(cfg, deployer)
    deploy.solve()

if __name__ == '__main__':
    main()
