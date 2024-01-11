#@Time      :2023/1/1
#@Author    :chenzhijian
#@Copyright   :Milab
#@For academic use only, commercial use strictly prohibited
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from attacks.ml_cw_pytorch import MLCarliniWagnerL2
from attacks.ml_rank1_pytorch import MLRank1
from attacks.ml_rank2_pytorch import MLRank2
from attacks.ml_deepfool_pytorch import MLDeepFool
from attacks.ml_lp import MLLP
# from attacks.ml_de_2007 import MLDE
# from attacks.ml_de_hideall import MLDE_hideall
from attacks.ml_jsma_pytorch import MLJSMA
from attacks.ml_jsmant_pytorch import MLJSMANT
from yolo_master.yolo_voc2012 import *
import numpy as np
import os
import math
from tqdm import tqdm
import logging
from advertorch import attacks
import torch
from torch import nn


tqdm.monitor_interval = 0
class AttackModel():
	def __init__(self, state):
		self.state = state
		self.y_target = state['y_target']
		self.y = state['y']
		self.data_loader = tqdm(state['data_loader'], desc='ADV')
		self.ori_loader = state['ori_loader']
		self.model = state['model']
		self.adv_save_x = state['adv_save_x']
		self.adv_batch_size = state['adv_batch_size']
		self.adv_begin_step = state['adv_begin_step']
		self.attack_model = None
		self.yolonet = YOLO()

	def attack(self):
		clip_min = 0.
		clip_max = 1.
		if self.state['adv_method'] == 'ml_cw':
			self.attack_model = MLCarliniWagnerL2(self.model)
			params = {'binary_search_steps': 10,
					  'y_target': None,
					  'max_iterations': 1000,
					  'learning_rate': 0.01,
					  'batch_size': self.adv_batch_size,
					  'initial_const': 1e5,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			self.ml_cw(params)
		elif self.state['adv_method'] == 'ml_rank1':
			self.attack_model = MLRank1(self.model)
			params = {'binary_search_steps': 10,
					  'y_target': None,
					  'max_iterations': 1000,
					  'learning_rate': 0.01,
					  'batch_size': self.adv_batch_size,
					  'initial_const': 1e5,
					  'clip_min': clip_min,
					  'clip_max': 1.}
			self.ml_rank1(params)
		elif self.state['adv_method'] == 'ml_rank2':
			self.attack_model = MLRank2(self.model)
			params = {'binary_search_steps': 10,
					  'y_target': None,
					  'max_iterations': 1000,
					  'learning_rate': 0.01,
					  'batch_size': self.adv_batch_size,
					  'initial_const': 1e5,
					  'clip_min': clip_min,
					  'clip_max': 1.}
			self.ml_rank2(params)
		elif self.state['adv_method'] == 'ml_deepfool':
			self.attack_model = MLDeepFool(self.model)
			params = {'y_target': True,
					  'max_iter': 20,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			self.ml_deepfool(params)
		elif self.state['adv_method'] == 'ml_de5' and self.state['target_type']!="hide_all":
			self.attack_model = MLDE(self.model)
			params = {'pop_size': 100,
					  'generation':200,
					  'clip_min': clip_min,
					  'clip_max': clip_max,
					  'eps':5,
					  'use_grad':0,
					  'batch_size': self.adv_batch_size,
					  'yolonet': self.yolonet}
			self.ml_de(params)
		elif self.state['adv_method'] == 'ml_de5' and self.state['target_type'] == "hide_all":
			self.attack_model = MLDE_hideall(self.model)
			params = {'pop_size': 100,
					  'generation': 200,
					  'clip_min': clip_min,
					  'clip_max': clip_max,
					  'eps': 5,
					  'use_grad': 0,
					  'batch_size': self.adv_batch_size,
					  'yolonet': self.yolonet}
			self.ml_de(params)
		elif self.state['adv_method'] == 'mla_lp':
		    self.attack_model = MLLP(self.model)
		    params = {'y_target': True,
		              'max_iter': 20,
		              'clip_min': clip_min,
		              'clip_max': clip_max}
		    self.mla_lp(params)
		elif self.state['adv_method'] == 'FGSM':
			params = {'y_target':self.y_target,
					  'eps': 0.3,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default: 0.03
			self.fgsm(params)
		elif self.state['adv_method'] == 'FGM':
			params = {'y_target':self.y_target,
					  'eps': 1.,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default: 0.03
			self.FGM(params)
		elif self.state['adv_method'] == 'MI-FGSM':
			params = {'y_target': self.y_target,
					  'eps': 0.3,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default: 0.3 - 40 - 0.01
			self.mi_fgsm(params)
		elif self.state['adv_method'] == 'BIM':
			params = {'y_target':self.y_target,
					  'eps': 0.1,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default: 0.1 - 10 - 0.05
			self.BIM(params)
		elif self.state['adv_method'] == 'PGD':
			params = {'y_target':self.y_target,
					  'eps': 0.3,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default: 0.3 - 40 - 0.01
			self.PGD(params)
		elif self.state['adv_method'] == 'SDA':
			params = {'y_target':self.y_target,
					  'eps': 10.00,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# default:0.3-40-0.01
			self.SDA(params)
		elif self.state['adv_method'] == 'LBFGS':
			params = {'y_target':self.y_target,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			self.LBFGS(params)
		elif self.state['adv_method'] == 'jsma':
			self.attack_model = MLJSMA(self.model)
			params = {'y_target': None,
					  'theta': 0.05,
					  'gamma': 8,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			# 伽马是大于0的整数，单位0.01%。也即gamma=100 扰动比例=100*0.01%=1%
			# 40-1500=0.1
			self.jsma(params)
		elif self.state['adv_method'] == 'jsmant':
			self.attack_model = MLJSMANT(self.model)
			params = {'y_target': True,
					  'theta': 0.015,
					  'gamma': 40,
					  'clip_min': clip_min,
					  'clip_max': clip_max}
			self.jsma_nt(params)
		elif self.state['adv_method'] == 'FFA':
			params = {'clip_min': clip_min,
					  'clip_max': clip_max}
			self.FFA(params)
		elif self.state['adv_method'] == 'zoo':
			params = {'clip_min': clip_min,
					  'clip_max': clip_max}
			self.zoo(params)
		else:
			print('please choose a correct adv method')

	def zoo(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
		A = A_pos + A_neg
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)
		for i, (input, target) in enumerate(self.data_loader):  # 这里的input是一整个batch的图片
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			adv = self.attack_model.generate_np(input[0].cpu().numpy(), A[begin:end], **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)



	def mla_lp(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
		A = A_pos + A_neg
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)
		for i, (input, target) in enumerate(self.data_loader):  # 这里的input是一整个batch的图片
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			adv = self.attack_model.generate_np(input[0].cpu().numpy(), A[begin:end], **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

	def ml_de(self, params):
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)
		diedai = 0
		for i, (input, target) in enumerate(self.data_loader):
			#input[0]是一个batch内的图像x，target是batch个标签
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			adv, count = self.attack_model.generate_np(input[0].cpu().numpy(),self.ori_loader, **params) #adv和输入格式一样就行
			diedai = diedai + count
			print('迭代次数：'+str(diedai))
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)
		# adv_list = []
		#
		# for i in range(begin_step, step):
		#     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
		#     tmp_file = np.load(tmp_file_path)
		#     adv_list.extend(tmp_file)
		# np.save(self.adv_save_x, np.asarray(adv_list))

	def jsma_nt(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
		A = A_pos + A_neg

		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)

		for i, (input, target) in enumerate(self.data_loader):
			# input[0]是一个batch内的图像x，target是batch个标签
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]

			adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
			# sys.exit()
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

	def jsma(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

		A = A_pos + A_neg
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)


		for i, (input, target) in enumerate(self.data_loader):
			# input[0]是一个batch内的图像x，target是batch个标签
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]

			adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
			# sys.exit()
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)


	def ml_cw(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)

		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]

			adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

		# adv_list = []
		#
		# for i in range(begin_step, step):
		#     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
		#     tmp_file = np.load(tmp_file_path)
		#     adv_list.extend(tmp_file)
		# np.save(self.adv_save_x, np.asarray(adv_list))

	def ml_rank1(self, params):
		y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)

		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			params['y_tor'] = y_tor[begin:end]
			adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

		# adv_list = []
		#
		# for i in range(begin_step, step):
		#     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
		#     tmp_file = np.load(tmp_file_path)
		#     adv_list.extend(tmp_file)
		# np.save(self.adv_save_x, np.asarray(adv_list))

	def ml_rank2(self, params):
		y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)

		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]
			params['y_tor'] = y_tor[begin:end]
			params['A_pos'] = A_pos[begin:end]
			params['A_neg'] = A_neg[begin:end]
			params['B_pos'] = B_pos[begin:end]
			params['B_neg'] = B_neg[begin:end]

			adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)

		# adv_list = []
		#
		# for i in range(begin_step, step):
		#     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
		#     tmp_file = np.load(tmp_file_path)
		#     adv_list.extend(tmp_file)
		# np.save(self.adv_save_x, np.asarray(adv_list))

	def ml_deepfool(self, params):
		_, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
		A = A_pos + A_neg

		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		print(params)

		for i, (input, target) in enumerate(self.data_loader):
			print('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			params['y_target'] = self.y_target[begin:end]

			adv = self.attack_model.generate_np(input[0].cpu().numpy(), A[begin:end], **params)
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)
		# adv_list = []
		#
		# for i in range(begin_step, step):
		#     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
		#     tmp_file = np.load(tmp_file_path)
		#     adv_list.extend(tmp_file)
		# np.save(self.adv_save_x, np.asarray(adv_list))

	def fgsm(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.GradientSignAttack(self.model,
												   loss_fn=MLLoss(),
												   eps=params['eps'],
												   clip_min=params['clip_min'],
												   clip_max=params['clip_max'],
												   targeted=True)


			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)

	def mi_fgsm(self, params):
		tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
		new_folder(tmp_folder_path)
		begin_step = self.adv_begin_step
		batch_size = self.adv_batch_size
		step = math.ceil(len(self.y_target) / batch_size)
		logging.info(params)

		class MLLoss(nn.Module):
			def __init__(self):
				super().__init__()
			def forward(self, o, y):
				o = torch.clamp(o, 1e-6, 1 - 1e-6)
				loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
				loss = loss.sum()
				return loss

		adversary = attacks.MomentumIterativeAttack(self.model,
													loss_fn=MLLoss(),
													eps=params['eps'],
													nb_iter=30,
													eps_iter=0.001,
													clip_min=params['clip_min'],
													clip_max=params['clip_max'],
													targeted=True)

		for i, (input, target) in enumerate(self.data_loader):
			logging.info('batch is {}'.format(i))
			logging.info('{} generator data, length is {}'.format(i, len(input[0])))
			if i < begin_step:
				continue
			params['batch_size'] = len(target)
			begin = i * batch_size
			end = begin + len(target)
			x = input[0].cpu().numpy()
			y_target = self.y_target[begin:end]
			y_target[y_target == -1] = 0

			x_t = torch.FloatTensor(x)
			y_target_t = torch.FloatTensor(y_target)
			if torch.cuda.is_available():
				x_t = x_t.cuda()
				y_target_t = y_target_t.cuda()
			x_t.requires_grad = True

			adv = adversary.perturb(x_t, y_target_t)
			adv = adv.detach().cpu().numpy()
			tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
			np.save(tmp_file_path, adv)



	def BIM(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.LinfBasicIterativeAttack(self.model,
														 loss_fn=MLLoss(),
														 eps=params['eps'],
														 nb_iter=30,
														 eps_iter=0.001,
														 clip_min=params['clip_min'],
														 clip_max=params['clip_max'],
														 targeted=True)

			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()

				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)

	def PGD(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.PGDAttack(self.model,
										  loss_fn=MLLoss(),
										  eps=params['eps'],
										  nb_iter=30,
										  eps_iter=0.001,
										  clip_min=params['clip_min'],
										  clip_max=params['clip_max'],
										  targeted=True)

			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)



	# PGD的 L1稀疏变体
	def SDA(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.SparseL1DescentAttack(self.model,
													  loss_fn=MLLoss(),
													  eps=300.00,
													  nb_iter=30,
													  eps_iter=30.0,
													  clip_min=params['clip_min'],
													  clip_max=params['clip_max'],
													  targeted=True)

			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)




	def FFA(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.FastFeatureAttack(self.model, loss_fn=None, eps=0.3, eps_iter=0.05,
                 nb_iter=10, rand_init=True, clip_min=0., clip_max=1.)


			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)



	# FGSM的无符号变体
	def FGM(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.GradientAttack(self.model,
											   loss_fn=MLLoss(),
											   eps=1.,
											   clip_min=params['clip_min'],
											   clip_max=params['clip_max'],
											   targeted=True)

			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				print(x.shape)
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True
				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)


	def LBFGS(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.LBFGSAttack(self.model,
											num_classes=20,
											batch_size=1,
											binary_search_steps=9,
											max_iterations=10,
											initial_const=1e-2,
											clip_min=0,
											clip_max=1,
											loss_fn = MLLoss(),
											targeted=True)


			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True

				x_t = torch.LongTensor(x_t.cpu().detach().numpy())
				y_target_t=torch.LongTensor(y_target_t.cpu().detach().numpy())


				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				x_t = x_t.to(device)
				y_target_t = y_target_t.to(device)


				adv = adversary.perturb(x_t, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)

	def JSMA(self, params):
			tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
			new_folder(tmp_folder_path)
			begin_step = self.adv_begin_step
			batch_size = self.adv_batch_size
			step = math.ceil(len(self.y_target) / batch_size)
			logging.info(params)

			class MLLoss(nn.Module):
				def __init__(self):
					super().__init__()
				def forward(self, o, y):
					o = torch.clamp(o, 1e-6, 1 - 1e-6)
					loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
					loss = loss.sum()
					return loss

			adversary = attacks.JacobianSaliencyMapAttack(self.model,
														  loss_fn=MLLoss(),
														  num_classes=20,
														  clip_min=0.0,
														  clip_max=1.0,
														  theta=1.0,
														  gamma=1.0,
														  comply_cleverhans=False)

			for i, (input, target) in enumerate(self.data_loader):
				logging.info('batch is {}'.format(i))
				logging.info('{} generator data, length is {}'.format(i, len(input[0])))
				if i < begin_step:
					continue
				params['batch_size'] = len(target)
				begin = i * batch_size
				end = begin + len(target)
				x = input[0].cpu().numpy()
				y_target = self.y_target[begin:end]
				y_target[y_target==-1] = 0

				x_t = torch.FloatTensor(x)
				y_target_t = torch.FloatTensor(y_target)
				if torch.cuda.is_available():
					x_t = x_t.cuda()
					y_target_t = y_target_t.cuda()
				x_t.requires_grad = True
				y = torch.FloatTensor(self.y)

				adv = adversary.perturb(y, y_target_t)
				adv = adv.detach().cpu().numpy()
				tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
				np.save(tmp_file_path, adv)



def new_folder(file_path):
	folder_path = os.path.dirname(file_path)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def get_target_set(y, y_target):
	y[y == 0] = -1
	A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
	A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
	B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
	B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0
	y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
	return y_tor, A_pos, A_neg, B_pos, B_neg
