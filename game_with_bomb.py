#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
"""Train the agent to collect coins and dodge bombs"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import numpy as np
import random
import copy
import time
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import scipy.io

# random.seed(11)


class Player():
	def __init__(self):
		self.P = {}
		self.Q = {}
		self.R = {}
		self.Ps = {}
		self.Trained = False

		self.mat_pre = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]
		self.mat_then = []
		self.mat_now = []
		self.code_pre = 0
		self.code_then = 0
		self.code_now = 0
		self.lastact = 0
		self.Posi = []
		self.trace = []

		self.PpairNum = []
		self.Pnum = 0
		self.Pnumlist = []
		self.S = []
		self.score = 0
		self.miss = 0
		self.gennum = 0

		self.coinnum = 0
		self.bombnum = 0
		self.hit = 0
		self.get = 0

	def LineGenerator(self):
		"""Generate a new line with bomb or coins"""
		# self.gennum+=1
		self.gennum = random.randint(1, 4)
		line = [0, 0, 0, 0]
		a = random.randint(1, 4)
		if self.gennum % 3 == 0:
			line[a - 1] = 2
			if self.Trained == True:
				self.bombnum += 1
		else:
			line[a - 1] = 1
			if self.Trained == True:
				self.coinnum += 1
		return line

	def GetCoding_pre(self):
		st = ''
		for i in range(len(self.mat_pre)):
			for j in range(len(self.mat_pre[0])):
				st = st + str(self.mat_pre[i][j])
		self.code_pre = st

	def GetCoding_then(self):
		st = ''
		for i in range(len(self.mat_then)):
			for j in range(len(self.mat_then[0])):
				st = st + str(self.mat_then[i][j])
		self.code_then = st

	def GetCoding_now(self):
		st = ''
		for i in range(len(self.mat_now)):
			for j in range(len(self.mat_now[0])):
				st = st + str(self.mat_now[i][j])
		self.code_now = st

	def move(self, penalty=-10):
		'''0-still, 1-left, 2-right ,to get P and R matrix'''
		s = random.randint(0, 2)
		self.GetCoding_pre()
		# print(s)
		posi = copy.deepcopy(self.mat_pre[3])
		pos = [0, 0, 0, 0]
		p = posi.index(1)
		if p == 0 and s == 1:
			pos = [1, 0, 0, 0]
		if p == 3 and s == 2:
			pos = [0, 0, 0, 1]
		if s == 0:
			pos = posi
		if s == 1 and p != 0:
			pos[p - 1] = 1
		if s == 2 and p != 3:
			pos[p + 1] = 1
		# print(pos)
		self.Posi = pos
		self.lastact = s

		NewLine = self.LineGenerator()
		self.mat_then = [NewLine, self.mat_pre[0], self.mat_pre[1], self.Posi]
		self.GetCoding_then()
		# print(self.code_pre)
		# print(self.code_then)
		# print(self.Posi)

		self.mat_pre = copy.deepcopy(self.mat_then)

		SS = (self.code_pre, self.code_then)
		SA = (self.code_pre, self.lastact)
		# print(SS)
		# print(SA)

		if self.code_pre not in self.S:
			self.S.append(self.code_pre)
		self.Pnumlist.append(len(self.S))

		if SA not in self.Ps.keys():
			self.Ps[SA] = {}
			self.R[SA] = {}

		if self.code_pre not in self.Q.keys():
			self.Q[self.code_pre] = {0: 0, 1: 0, 2: 0}
		# self.Qr=copy.deepcopy(self.Q)

		# print(len(self.Ps.keys()))
		self.PpairNum.append(len(self.Ps.keys()))

		try:
			self.Ps[SA][self.code_then] = self.Ps[SA][self.code_then] + 1
		except KeyError:
			self.Ps[SA][self.code_then] = 0
			self.Ps[SA][self.code_then] = self.Ps[SA][self.code_then] + 1

		if self.mat_then[2][self.mat_then[3].index(1)] == 1:
			self.R[SA][self.code_then] = 10
		else:
			if self.mat_then[2][self.mat_then[3].index(1)] == 2:
				self.R[SA][self.code_then] = penalty
			else:
				self.R[SA][self.code_then] = 0

	def GetP(self):
		self.P = copy.deepcopy(self.Ps)
		for sa in self.Ps.keys():
			total = sum(self.Ps[sa].values())
			for s in self.Ps[sa].keys():
				self.P[sa][s] = self.P[sa][s] / total

	def GetR(self):
		for sa in self.P.keys():
			tpe = 0
			for s in self.P[sa].keys():
				tpe = tpe + self.P[sa][s] * self.R[sa][s]
			self.R[sa] = tpe
		print(len(self.R.keys()))

	def Train(self, num=20000, gamma=0.8):
		"""Train the agent for given steps and discount factor gamma"""
		for tt in range(num):
			s = random.randint(0, 2)
			self.GetCoding_pre()
			# print(s)
			posi = copy.deepcopy(self.mat_pre[3])
			pos = [0, 0, 0, 0]
			p = posi.index(1)
			if p == 0 and s == 1:
				pos = [1, 0, 0, 0]
			if p == 3 and s == 2:
				pos = [0, 0, 0, 1]
			if s == 0:
				pos = posi
			if s == 1 and p != 0:
				pos[p - 1] = 1
			if s == 2 and p != 3:
				pos[p + 1] = 1
			# print(pos)
			self.Posi = pos
			self.lastact = s

			NewLine = self.LineGenerator()
			self.mat_then = [
				NewLine,
				self.mat_pre[0],
				self.mat_pre[1],
				self.Posi]
			self.GetCoding_then()
			# print(self.code_pre)
			# print(self.code_then)
			# print(self.Posi)

			self.mat_pre = copy.deepcopy(self.mat_then)

			SS = (self.code_pre, self.code_then)
			SA = (self.code_pre, self.lastact)

			ex = 0
			for sta in self.P[(self.code_pre, self.lastact)].keys():
				ex = ex + self.P[(self.code_pre, self.lastact)
								 ][sta] * max(self.Q[sta].values())

			ex = ex * gamma
			self.Q[self.code_pre][self.lastact] = self.R[(
				self.code_pre, self.lastact)] + ex
		self.Trained = True

	def Runn(self, time=3):
		for tt in range(time):
			NewLine = self.LineGenerator()
			self.mat_now = [
				NewLine,
				self.mat_pre[0],
				self.mat_pre[1],
				self.mat_pre[3]]
			# print(1)
			# print(np.array(self.mat_now))
			if self.mat_now[2][self.mat_now[3].index(1)] == 1:
				self.score = self.score + 1
				self.get += 1
			if self.mat_now[2][self.mat_now[3].index(1)] == 2:
				self.score = self.score - 1
				self.hit += 1
			self.trace.append(np.array(self.mat_now))
			self.GetCoding_now()
			if self.Trained == True:
				s = max(self.Q[self.code_now], key=self.Q[self.code_now].get)
			else:
				s = random.randint(0, 2)

			posi = copy.deepcopy(self.mat_now[3])
			pos = [0, 0, 0, 0]
			p = posi.index(1)
			if p == 0 and s == 1:
				pos = [1, 0, 0, 0]
			if p == 3 and s == 2:
				pos = [0, 0, 0, 1]
			if s == 0:
				pos = posi
			if s == 1 and p != 0:
				pos[p - 1] = 1
			if s == 2 and p != 3:
				pos[p + 1] = 1
			self.Posi = pos
			self.lastact = s
			self.mat_now[3] = self.Posi
			# print(np.array(self.mat_now))
			self.mat_pre = copy.deepcopy(self.mat_now)
		self.CCR = self.score / time


class Ball:
	def __init__(self, canvas, color):
		self.canvas = canvas
		self.color = color

	def draw(self, t):
		if self.color == 'blue':
			self.id = canvas.create_rectangle(20, 20, 70, 70, fill=self.color)
		else:
			self.id = canvas.create_oval(20, 20, 80, 80, fill=self.color)
		self.canvas.move(self.id, t[1], t[0])


tk = Tk()
tk.title("maze")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=400, height=300, bd=0, highlightthickness=0)

canvas.pack()
im = PhotoImage(file='coin.gif')
canvas.create_image(0, 0, anchor=NW, image=im)
tk.update()

def main():
	CCR = []
	# for tt in range(-15,5):
	tt = -10
	print(tt)
	random.seed(11)

	ob = Player()

	for i in range(350000):
		ob.move(tt)

	ob.GetP()
	ob.GetR()
	print('Start training?')
	hold=input()
	ob.Train(100000, 0.8)
	# print(ob.Q)
	ob.Runn(2000)
	# print(ob.score)
	CCR.append(ob.score)

	# print(ob.bombnum)
	# print(ob.coinnum)

	# print(ob.Pnumlist)
	# print(ob.PpairNum)
	print(ob.score)
	print(ob.get)
	print(ob.hit)

	#scipy.io.savemat('g1_MC.mat',{'CCR': CCR})
	#load_data = scipy.io.loadmat('yu.mat')


	cord = {
		(
			0, 0): [
				0, 0], (0, 1): [
					0, 100], (0, 2): [
						0, 200], (0, 3): [
							0, 300], (1, 0): [
								100, 0], (1, 3): [
									100, 300], (1, 1): [
										100, 100], (1, 2): [
											100, 200], (2, 0): [
												200, 0], (2, 1): [
													200, 100], (2, 2): [
														200, 200], (2, 3): [
															200, 300], (3, 0): [
																200, 0], (3, 1): [
																	200, 100], (3, 2): [
																		200, 200], (3, 3): [
																			200, 300]}

	for mat in ob.trace:

		pt = np.nonzero(mat)
		# print(pt)

		ball1 = Ball(canvas, 'yellow')
		ball2 = Ball(canvas, 'yellow')
		ball3 = Ball(canvas, 'yellow')
		ball4 = Ball(canvas, 'blue')
		ballb1 = Ball(canvas, 'black')
		ballb2 = Ball(canvas, 'black')
		ballb3 = Ball(canvas, 'black')
		ballb4 = Ball(canvas, 'blue')

		ball = [ball1, ball2, ball3, ball4]
		ballb = [ballb1, ballb2, ballb3, ballb4]

		ls = []
		for i in range(pt[0].shape[0]):
			if mat[pt[0][i], pt[1][i]] == 1:
				ball[i].draw(cord[(pt[0][i], pt[1][i])])
			else:
				ballb[i].draw(cord[(pt[0][i], pt[1][i])])

		# ball.draw(ls)

		tk.update_idletasks()
		tk.update()
		time.sleep(0.5)

		for i in range(pt[0].shape[0]):
			try:
				ball[i].canvas.delete(ball[i].id)
			except BaseException:
				continue
		for i in range(pt[0].shape[0]):
			try:
				ballb[i].canvas.delete(ballb[i].id)
			except BaseException:
				continue


#scipy.io.savemat('yu.mat',{'yuan': yu})
#load_data = scipy.io.loadmat('sa.mat')
# sa=load_data['sa']

if __name__ == '__main__':
    main()


tt = [i for i in range(350000)]

'''plt.figure()

plt.plot(tt,ob.PpairNum,color='blue',label='with bomb')
plt.plot(tt,sa[0],color='green',label='without bomb')
#plt.plot(nullrate,FSC,color='red',linestyle='-.',label='CCR: Pearson Correlation')
#plt.plot(nullrate,FSC2,color='blue',linestyle='-.',label='CCR: Partial Cosine')

plt.xlabel("Episode number")
plt.ylabel("S-A pair Number")
plt.title("states expansion")
plt.grid(True)
plt.legend()
plt.show()'''
