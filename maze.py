#!usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
"""Train an agent to run the maze. It should first collect the nut in the maze"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


from PIL import Image, ImageTk
from tkinter import *
import time
import random
import numpy as np
import sys


def rand_index(pmflist):
    """choose an index given transition probability"""
    pmflist = list(pmflist)
    for i in range(len(pmflist)):
        pmflist[i] = int(pmflist[i] * 100)
    summ = sum(pmflist)
    randnum = random.randint(1, summ - 1)
    flag = 0
    ind = 0
    while ind < len(pmflist):
        flag = flag + pmflist[ind]
        if flag > randnum:
            re = ind
            break
        ind = ind + 1
    return re


class Point():
    """an agent object"""

    def __init__(self, posi, R, Q, T):
        self.posi = posi  # position
        self.R = R  # reward matrix
        self.Q = Q  # Q matrix
        self.T = T  # Probability Transition Matrix
        self.step = 0  # record the steps
        self.trained = False

    def simplemove(self):
        """if not trained, using P to random walk; if trained, using Bellman Optimal V"""
        if self.trained == True:
            tplist = list(self.Q[self.posi])
            nex = tplist.index(max(tplist))
            self.posi = nex

        else:
            nex = rand_index(self.T[self.posi])
            self.posi = nex
            self.step = self.step + 1

    def move(self):
        """moving method used in training"""
        nex = rand_index(T[self.posi])
        self.Q[self.posi][nex] = self.R[self.posi][nex] + \
            0.8 * max(Q[nex])  # Bellman Opt Q recursion
        self.posi = nex

    def sett(self):
        """renewal to the initial condition"""
        self.posi = 0
        self.step = 0

    def train(self, times):
        """train the agent in certain ellipsoids"""
        for i in range(times):
            self.sett()
            nn = 0
            while True:
                nn = nn + 1
                self.move()
                ball = Ball(canvas, 'red')
                ball.draw(cord[self.posi])
                tk.update_idletasks()
                tk.update()
                ball.canvas.delete(ball.id)
                if self.posi == 7:
                    break
                if self.posi == 14:
                    break
        self.trained = True


T = np.zeros([15, 15], dtype=float)
T[0][1] = 1
T[1][2] = 0.5
T[1][5] = 0.5
T[2][1] = 0.5
T[2][3] = 0.5
T[3][2] = 0.5
T[3][4] = 0.5
T[4][3] = 0.5
T[4][7] = 0.5
T[5][1] = 0.5
T[5][6] = 0.5
T[6][9] = 1
T[7][7] = 1
T[8][9] = 1
T[9][8] = 0.5
T[9][10] = 0.5
T[10][9] = 0.5
T[10][11] = 0.5
T[11][12] = 0.5
T[11][10] = 0.5
T[12][11] = 0.5
T[12][13] = 0.5
T[13][12] = 0.5
T[13][14] = 0.5
T[14][14] = 1

cord = {
    0: [
        380, 370], 1: [
            270, 370], 10: [
                270, 370], 2: [
                    270, 190], 11: [
                        270, 190], 3: [
                            270, 30], 12: [
                                270, 30], 4: [
                                    140, 30], 13: [
                                        140, 30], 6: [
                                            140, 190], 8: [
                                                140, 190], 5: [
                                                    140, 370], 9: [
                                                        140, 370], 7: [
                                                            10, 30], 14: [
                                                                10, 30]}


print(T)

R = np.zeros([15, 15], dtype=float)
R = R - 1
for i in range(15):
    for j in range(15):
        if T[i][j] != 0:
            R[i][j] = 0
R[4][7] = 100
R[5][6] = 50
R[13][14] = 100

Q = np.zeros((15, 15), dtype=float)  # initialize Q matrix


class Ball:
    def __init__(self, canvas, color):
        self.canvas = canvas

    def draw(self, tu):
        self.id = canvas.create_oval(20, 20, 80, 80, fill='white')
        self.canvas.move(self.id, tu[1], tu[0])


print(1)

tk = Tk()
tk.title("maze")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=500, height=500, bd=0, highlightthickness=0)

canvas.pack()
im = PhotoImage(file='mazee.gif')
canvas.create_image(0, 0, anchor=NW, image=im)
tk.update()


def main():
    agent = Point(0, R, Q, T)  # build the agent with start state 0

    trace_1 = []
    trace1 = [0]
    # for i in range(100):
    while True:
        agent.simplemove()
        trace1.append(agent.posi)  # record the random walk trace
        if agent.posi == 7:  # end the ellipsoid if reach state 10
            break
        if agent.posi == 14:  # end the ellipsoid if reach state 10
            break
    print(trace1)

    print('Start training?')
    tp = input()

    start_time = time.time()
    agent.train(200)  # train the agent for 200 ellipsoid
    print(time.time() - start_time)  # training time
    # print(agent.Q)
    agent.sett()

    trace = [0]
    te = 0
    while True:
        agent.simplemove()
        trace.append(agent.posi)
        if agent.posi == 7:  # record the optimal trace
            break
        if agent.posi == 14:  # record the optimal trace
            break
        if te > 40:
            break
        te = te + 1
    print(trace)
    # print(agent.Q) # return Q* matrix
    while True:
        for i in range(len(trace)):
            ball = Ball(canvas, 'red')
            ball.draw(cord[trace[i]])

            tk.update_idletasks()
            tk.update()
            time.sleep(0.2)
            ball.canvas.delete(ball.id)


if __name__ == '__main__':
    main()
