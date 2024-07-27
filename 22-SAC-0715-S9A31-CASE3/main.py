import numpy as np

from utils import evaluate_policy, str2bool
from datetime import datetime
from SACD import SACD_agent
from env import Env
import os, shutil
import argparse
import torch
import random

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=2e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e4  , help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[128,128], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
opt = parser.parse_args()
print(opt)

def main():
	#Create Env
	EnvName = 'case4'
	env = Env()
	eval_env = Env()
	opt.state_dim = env.n_states
	opt.action_dim = env.n_actions
	opt.max_e_steps = env._max_episode_steps

	# Seed Everything
	# env_seed = opt.seed
	# torch.manual_seed(opt.seed)
	# torch.cuda.manual_seed(opt.seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	# print("Random Seed: {}".format(opt.seed))

	print('Algorithm: SACD','  Env:',EnvName,'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, 'reward_weight:',env.reward_weight,'\n')

	if opt.write:
		from torch.utils.tensorboard import SummaryWriter
		timenow = str(datetime.now())[0:-10]
		timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
		writepath = 'runs/SACD_{}'.format(EnvName) + timenow
		if os.path.exists(writepath): shutil.rmtree(writepath)
		writer = SummaryWriter(log_dir=writepath)

	#Build model
	if not os.path.exists('model'): os.mkdir('model')
	agent = SACD_agent(**vars(opt))
	if opt.Loadmodel: agent.load(opt.ModelIdex, EnvName)

	if opt.render:
		for i in range(10):
			evaluate_policy(env, agent, 1,True)
			#print('EnvName:', EnvName, 'seed:', opt.seed, 'score:', score)
	else:
		total_steps = 0
		while total_steps < opt.Max_train_steps:
			# s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
			# env_seed += 1
			env.reset()
			s = env.INITSTATE
			done = False

			'''Interact & trian'''
			while not done:
				#e-greedy exploration
				if total_steps < opt.random_steps: a = np.random.randint(0, env.n_actions)
				else: a = agent.select_action(s, deterministic=False)
				s_next, r, dw = env.step(a) # dw: dead&win; tr: truncated
				done = dw
				agent.replay_buffer.add(s, a, r, s_next, dw)
				s = s_next

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						agent.train()

				'''record & log'''
				if total_steps % opt.eval_interval == 0:
					score,target ,actions= evaluate_policy(eval_env, agent, turns=3)
					if opt.write:
						writer.add_scalar('ep_r', score, global_step=total_steps)
						writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
						writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
					print(
						  'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score),'target:',target,'actions:',actions)
				total_steps += 1

				'''save model'''
				if total_steps % opt.save_interval == 0:
					agent.save(int(total_steps/1000), EnvName)
	env.close()
	eval_env.close()


if __name__ == '__main__':
	main()

