import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import Double_Q_Net, Policy_Net, ReplayBuffer


class SACD_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005
		self.H_mean = 0
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

		self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

		if self.adaptive_alpha:
			# 我们使用 0.6，因为推荐的 0.98 会导致 alpha explosion.
			self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

	def select_action(self, state, deterministic):
		with torch.no_grad():

			# state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc) #from (s_dim,) to (1, s_dim)
			# 假设 state 是一个列表，需要先将其转换为 NumPy 数组
			state_np = np.array(state)

			# 将 NumPy 数组转换为 PyTorch 张量，并添加一个额外的维度在第一维度上
			state = torch.FloatTensor(state_np[np.newaxis, :]).to(self.dvc)

			# 现在 state_tensor 是一个形状为 (1, s_dim) 的张量，可以在需要的地方使用它
			probs = self.actor(state)

			if deterministic:
				a = probs.argmax(-1).item()
			else:
				a = Categorical(probs).sample().item()


			return a

	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#------------------------------------------ Train Critic ----------------------------------------#
		'''Compute the target soft Q value'''
		with torch.no_grad():
			next_probs = self.actor(s_next) #[b,a_dim]
			next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]
			next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
			target_Q = r + (~dw) * self.gamma * v_next

		'''Update soft Q net'''
		q1_all, q2_all = self.q_critic(s) #[b,a_dim]
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b,1]
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#------------------------------------------ Train Actor ----------------------------------------#
		for params in self.q_critic.parameters():
			#Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
			params.requires_grad = 	False

		probs = self.actor(s) #[b,a_dim]
		log_probs = torch.log(probs + 1e-8) #[b,a_dim]
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)  #[b,a_dim]
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=True) #[b,1]

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	True

		#------------------------------------------ Train Alpha ----------------------------------------#
		if self.adaptive_alpha:
			with torch.no_grad():
				self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
			alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp().item()

		#------------------------------------------ Update Target Net ----------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, timestep, EnvName):
		torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}_{EnvName}.pth")
		torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}_{EnvName}.pth")


	def load(self, timestep, EnvName):
		self.actor.load_state_dict(torch.load(f"./model/sacd_actor_{timestep}_{EnvName}.pth", map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load(f"./model/sacd_critic_{timestep}_{EnvName}.pth", map_location=self.dvc))
