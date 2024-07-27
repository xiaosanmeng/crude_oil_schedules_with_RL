

import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Env():

    def __init__(self):

        # TKi=[编号，类型，容量，存量]
        self.TK = [
            [1, 5, 16000, 8000],
            [2, 5, 34000, 30000],
            [3, 4, 34000, 30000],
            [4, 0, 34000, 0],
            [5, 3, 34000, 30000],
            [6, 1, 16000, 16000],
            [7, 6, 20000, 16000],
            [8, 6, 16000, 5000],
            [9, 0, 16000, 0],
            [10, 0, 30000, 0]
        ]
        # RT=[编号，炼油类型，炼油量,炼油类型，炼油量]
        self.RT = [
            [1, 5, 38000, 1, 41996],
            [2, 6, 21000, 2, 49000],
            [3, 4, 30000, 3, 120000]
        ]
        # 初始化状态
        #0·1·2：[the completion rate of each refining tower]
        #3-12:每个罐的存有情况%



        self.INITSTATE =[0,   0,   0,
                         0,   round(30/34,2),   round(30/34,2),  0,   round(30/34,2),
                         1,   round(16/20,2),   round(5/16,2),  0,   0
            ]

        #罐日志，数组下标+1=罐编号，元素为[chargeTIME,refinetime,cot1,cot2]
        self.log_tank = [[0, 0] for _ in range(len(self.TK))]
        #常量
        self.RESIDENCE_TIME = 6
        self.F_SDU = [333.3, 291.7, 625]
        self.Mt = [[0, 11, 12, 13, 10, 15],
              [11, 0, 11, 12, 13, 10],
              [12, 11, 0, 10, 12, 13],
              [13, 12, 10, 0, 11, 12],
              [10, 13, 12, 11, 0, 11],
              [15, 10, 13, 12, 11, 0]]
        self.Mp = [[0, 11, 12, 13, 7, 15],
              [10, 0, 9, 2, 13, 7],
              [13, 8, 0, 7, 2, 13],
              [13, 12, 7, 0, 11, 12],
              [7, 13, 12, 11, 0, 11],
              [15, 7, 13, 12, 11, 0]]
        # 是否为安全状态
        self.SecureState = True
        # 两个计划
        self.schedule_distiller = [[], [], []]  # [TANK,COT,V,START,END]
        self.schedule_pipe = []  # [TANK,COT,V,START,END,(RATE]
        # 辅助
        self.time_ODT = 0
        self.time_ODF = [0, 0, 0]


        #优化目标
        self.target=[0,0,0,0]
        self.preTarget=[0,0,0,7]

        self.n_states = len(self.INITSTATE)
        self.n_actions = 11
        self.refine()  # 预调度

    def reset(self):
        self.__init__()

    def step(self, action,render=False,rendertime=1):
        #塔：1.最急迫，  2.最不紧迫
        #罐：1.体积最小  2.体积最大  3.体积最接近   4最近释放 5最早释放
        #泵速 1.nil     2.1250
        #return : state reward done
        #***************************************字典映射***************************************************************
        # 创建动作列表
        actionsList = [111, 112, 122, 132, 212,
                       222, 232, 142, 242, 152,
                       252]

        # 创建字典
        action_dict = {i: action for i, action in enumerate(actionsList)}
        action=action_dict[action]

        #操作定义
        rate= 0 if (action%10)==1 else 1250
        disitller=self.selectDistiller(action//100) if rate!=0 else None
        tank=self.selectTank((action%100)//10,disitller) if rate!=0 else None

        #执行调度
        self.charge(disitller,tank,rate)
        self.refine()
        self.isSecureState()
        # ++++++++++++++++++++++++++++++++done++++++++++++++++++++++++++++++++++++++++++
        done = ((self.RT[0][2] + self.RT[0][4] + self.RT[1][2] + self.RT[1][4] + self.RT[2][2] + self.RT[2][
            4]) == 0 or not self.SecureState)


        if(done and self.SecureState):
            self.time_ODT=240




        if (self.SecureState):
            # ++++++++++++++++++++++++++++++++reward_mix_pipe++++++++++++++++++++++++++++++++++++++++++
            cot_list = []
            for i in range(len(self.schedule_pipe)):
                if (self.schedule_pipe[i][1] != 0):
                    cot_list.append(self.schedule_pipe[i][1])
            for i in range(len(cot_list) - 1):
                pipe_cot_x = cot_list[i]
                pipe_cot_y = cot_list[i + 1]
                self.target[0] += self.Mp[pipe_cot_x - 1][pipe_cot_y - 1]





            # ++++++++++++++++++++++++++++++++reward_mix_tank++++++++++++++++++++++++++++++++++++++++++

            for log_tank_i in self.log_tank:
                for i in range(len(log_tank_i) - 3):
                    self.target[1] += self.Mt[log_tank_i[2 + i][0] - 1][log_tank_i[3 + i][0]  - 1]



            # ++++++++++++++++++++++++++++++++reward_change_tank++++++++++++++++++++++++++++++++++++++++++


            for schedule_distiller_i in self.schedule_distiller:
                self.target[2]+=len(schedule_distiller_i)



            # ++++++++++++++++++++++++++++++++reward_used_tank++++++++++++++++++++++++++++++++++++++++++



            for log_tanki in self.log_tank:
                if (len(log_tanki) > 2):
                    self.target[3]+= 1
            #++++++++++++++++++++++++++++++++++++REWARD
            reward=sum(self.preTarget)-sum(self.target)
            #reward=self.preTarget[3]-self.target[3]



            self.preTarget=self.target
            self.target=[0,0,0,0]

        else:
            reward=-200




        # ++++++++++++++++++++++++++++++++State计算++++++++++++++++++++++++++++++++++++++++++
        state=[ 0 for _ in range(13)]
        for i in range(3):
            state[i] = round(1 - ((self.RT[i][2] + self.RT[i][4]) / (240 * self.F_SDU[i])),2)


        for i in range(len(self.TK)):
            if(self.log_tank[i][1]<=(self.time_ODT) and self.TK[i][3]==0):
                state[i+3]=0
            elif(self.TK[i][3]==0):

                state[i+3]=round(self.log_tank[i][-1][1]/self.TK[i][2],2)
            else:
                state[i + 3] = round(self.TK[i][3] / self.TK[i][2], 2)










        if render:
            self.render(rendertime)

        return state,reward,done

    def render(self,renderTime=1):
        #[TANK, COT, V, START, END]
        data = [self.schedule_distiller[0],
                self.schedule_distiller[1],
                self.schedule_distiller[2],
                self.schedule_pipe]
        color = {0:'w',
                 1:'b',
                 2:'g',
                 3:'r',
                 4:'y',
                 5:'c',
                 6:'m'}
        # 画布设置，大小与分辨率
        plt.figure(figsize=(10, 4), dpi=80)
        plt.xlim(0,240)
        plt.ylim(0,5)
        plt.xticks(range(0, 241, 10))
        plt.yticks(range(0,6,1))

        #甘特图绘制
        for i in range(len(data)):
            for j in range(len(data[i])):
                plt.barh(y=4-i, width=(data[i][j][4]-data[i][j][3]),height=0.5, left=data[i][j][3], color=color[data[i][j][1]],edgecolor='k')
                plt.text(y=4-i,x=(data[i][j][4]+data[i][j][3])/2,s=("TK"+str(data[i][j][0]) if data[i][j][0]!=0 else ""),ha='center', va='center', color='w',fontsize=15)
        #其他设置
        #plt.grid(linestyle="--", alpha=0.5)
        # # XY轴标签
        plt.xlabel("调度时间/s")
        plt.ylabel("蒸馏塔/管道调度信息")

        plt.show(block=False)
        plt.pause(renderTime)
        plt.close()

    def refine(self):
        # 根据state的信息进行炼油，更改env信息
        for distiller in self.RT:
            # 该蒸馏塔a任务未完成
            if (distiller[2] > 0):
                # 遍历罐列表
                for tank in self.TK:
                    # 类型相同，存量不为0,并且可用
                    if (distiller[2] > 0 and tank[1] == distiller[1] and tank[3] != 0 and self.log_tank[tank[0] - 1][
                        0] <= self.time_ODF[distiller[0] - 1]):

                        oilType = tank[1]
                        refineVolume = tank[3]
                        if (len(self.schedule_distiller[distiller[0] - 1]) != 0):
                            startTime = self.schedule_distiller[distiller[0] - 1][-1][4]
                        else:
                            startTime = 0
                        endTime = round(startTime + (refineVolume / self.F_SDU[distiller[0] - 1]),1)
                        schedule = [tank[0], oilType, refineVolume, startTime, endTime]
                        self.schedule_distiller[distiller[0] - 1].append(schedule)
                        self.time_ODF[distiller[0] - 1] = endTime

                        # ---------------------------------------------------------------
                        self.log_tank[tank[0] - 1].append([oilType,refineVolume])
                        self.log_tank[tank[0] - 1][1] = endTime  # refine time更新
                        self.TK[tank[0] - 1][3] = 0
                        self.RT[distiller[0] - 1][2] -= refineVolume
            # 该蒸馏塔b任务未完成
            if (distiller[4] > 0):
                # 遍历罐列表
                for tank in self.TK:
                    # 类型相同，存量不为0,并且可用
                    if (distiller[4] > 0 and tank[1] == distiller[3] and tank[3] != 0 and self.log_tank[tank[0] - 1][
                        0] <= self.time_ODF[distiller[0] - 1]):
                        oilType = tank[1]
                        refineVolume = tank[3]
                        if (len(self.schedule_distiller[distiller[0] - 1]) != 0):
                            startTime = self.schedule_distiller[distiller[0] - 1][-1][4]
                        else:
                            startTime = 0
                        endTime = round(startTime + (refineVolume / self.F_SDU[distiller[0] - 1]),1)
                        schedule = [tank[0], oilType, refineVolume, startTime, endTime]
                        self.schedule_distiller[distiller[0] - 1].append(schedule)
                        self.time_ODF[distiller[0] - 1] = endTime

                        # ---------------------------------------------------------------
                        self.log_tank[tank[0] - 1].append([oilType,refineVolume])
                        self.log_tank[tank[0] - 1][1] = endTime  # refine time更新
                        self.TK[tank[0] - 1][3] = 0
                        self.RT[distiller[0] - 1][4] -= refineVolume

    def charge(self,distiller,tank,rate):
        # parameter：蒸馏塔实际编号，充油罐实际编号，充油速率
        # 根据传入参数进行充油，更改env信息

        # tank=[编号，类型，容量，存量]  RT=[编号，炼油类型，炼油量,炼油类型，炼油量]
        #特殊判断
        if(tank==None or distiller==None):
            rate=0
        # ==================计算开始时间=================================================
        if (len(self.schedule_pipe) != 0):  # 如果不为空，求出上一个管道运输信息[TANK,COT,V,START,END]
            startTime = self.schedule_pipe[-1][4]  # -1可定位到最后一个元素
        else:
            startTime = 0
        #===============================================================================
        if (rate != 0):
            # 计算转运体积&类型，分两种情况：炼油类型1＆2
            if (self.RT[distiller - 1][2] != 0):  # 炼油任务1未完成
                volume = self.TK[tank - 1][2] if self.RT[distiller - 1][2] > self.TK[tank - 1][
                    2] else self.RT[distiller - 1][2]
                cot = self.RT[distiller - 1][1]

            else:
                volume = self.TK[tank - 1][2] if self.RT[distiller - 1][4] > self.TK[tank - 1][
                    2] else self.RT[distiller - 1][4]
                cot = self.RT[distiller - 1][3]
            if (volume != 0):
                # 得出计划i
                schdule = [tank, cot, volume, startTime, startTime + volume / rate, rate]
                self.schedule_pipe.append(schdule)
                # 更新time_ODT
                self.time_ODT = round(startTime + volume / rate,1)
                # 充油
                self.TK[tank - 1][1] = cot
                self.TK[tank - 1][3] = volume
                self.log_tank[tank - 1][0] = round(startTime + volume / rate + self.RESIDENCE_TIME,1)

        else:  # 停运
            # 两种情况：endtime_a and endTime_b

            urge_time = min(self.time_ODF)
            endTime_b = endTime_a = 240
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~空罐统计~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            emptyTK=[]
            for i in range(len(self.TK)):
                if (self.TK[i][3] == 0 and self.log_tank[i][1] <= self.time_ODT):
                    emptyTK.append(self.TK[i])
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #~~~~~~~~~~~~~~~~~~~~~~~~~存在空罐，对应time_a~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (len(emptyTK) != 0):
                mintank_v = min(tank[2] for tank in emptyTK)
                endTime_a = urge_time - mintank_v / 1250 - self.RESIDENCE_TIME
            #~~~~~~~~~~~~~~~~~~~~~~~~~不存在空罐需要等待释放，对应time_b~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in range(len(self.TK)):
                if (self.TK[i][3] == 0 and self.log_tank[i][1] < endTime_b and self.log_tank[i][1] > startTime):  # 存量为0 and refineTime<240 and refineTime>startTime
                    endTime_b = self.log_tank[i][1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~time_a,time_b取最小值~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            endTime = min(endTime_a, endTime_b)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~边界处理！！！！由secure处理！！！~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (endTime < startTime):
                endTime = startTime
            #schedule
            schdule = [0, 0, 0, startTime, endTime, rate]
            self.schedule_pipe.append(schdule)
            # 更新time_ODT
            self.time_ODT = endTime

    def selectDistiller(self,type):

        #parameter：type:1.最急迫，  2.最宽裕， 3.居中（无居中情况时，采用最紧迫
        #return：   distiller的实际编号
        #》》》》》》》》》》》》》》》》》》》》最紧迫，无边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        distiller = None
        if type==1:
            distiller=self.time_ODF.index(min(self.time_ODF))+1
        # 》》》》》》》》》》》》》》》》》》》》最宽裕，存在边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        if type==2:
            max=0
            for i in range(len(self.time_ODF)):
                if(math.ceil(self.time_ODF[i])<240 and self.time_ODF[i]>max):
                    max=self.time_ODF[i]
                    distiller=i+1
        #》》》》》》》》》》》》》》》》》》》》居中，存在边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        if type==3:
            count=0#非240 的元素,count==3的情况下才有居中情况
            for i in range(len(self.time_ODF)):
                if (math.ceil(self.time_ODF[i])<240):
                    count+=1
            if(count==3):
                max=0
                second=0
                for i in range(len(self.time_ODF)):
                    if(self.time_ODF[i]>max):
                        max=self.time_ODF[i]

                for i in range(len(self.time_ODF)):
                    if(self.time_ODF[i]<max and self.time_ODF[i]>second):
                        second=self.time_ODF[i]
                        distiller=i+1
            else:
                distiller=self.time_ODF.index(min(self.time_ODF))+1
        return distiller

    def selectTank(self,type,distiller):
        #parameter:     type: 1.体积最小  2.体积最大  3.体积最接近
        #               distiller :蒸馏塔的实际编号
        #return:        tank的实际编号‘
        #********************************空罐判断*****************************************
        emptyTK=[]
        for tank in self.TK:
            if(tank[3]==0 and self.log_tank[tank[0]-1][1]<=self.time_ODT):
                emptyTK.append(tank)
        tank = None

        # ********************************1.体积最小 *****************************************
        if type==1:
            minV=float("inf")
            for emptyTank in emptyTK:
                if emptyTank[2]<minV:
                    minV=emptyTank[2]
                    tank=emptyTank[0]
        # ********************************2.体积最大 *****************************************
        if type==2:
            maxV=0
            for emptyTank in emptyTK:
                if emptyTank[2]>maxV:
                    maxV=emptyTank[2]
                    tank=emptyTank[0]
        # ********************************3.体积最接近 *****************************************
        if type==3:
            #体积最接近：need_v - tank_v  的值最小，同时应尽可能地保证diff<0
            needVolume=self.RT[distiller-1][2] if self.RT[distiller-1][2]!=0 else self.RT[distiller-1][4]
            diff=[needVolume,float("-inf")]#[diff>=0,diff<0]
            for emptyTank in emptyTK:
                if needVolume-emptyTank[2]>0 and needVolume-emptyTank[2]<diff[0]:
                    diff[0]=needVolume-emptyTank[2]
                    tank=emptyTank[0]
                if needVolume-emptyTank[2]<0 and needVolume-emptyTank[2]>diff[1]:
                    diff[1]=needVolume-emptyTank[2]
                    tank=emptyTank[0]

        # ********************************4.最近释放 *****************************************
        if type==4:
            releaseTime=0
            for i in range(len(self.TK)):
                if(self.TK[i][3]==0 and self.log_tank[i][1]<=self.time_ODT and self.log_tank[i][1]>releaseTime):
                    releaseTime=self.log_tank[i][1]
                    tank=self.TK[i][0]

        # ********************************5.最早释放 *****************************************
        if type == 5:
            releaseTime = 240
            for i in range(len(self.TK)):
                if(self.TK[i][3]==0 and self.log_tank[i][1]<=self.time_ODT and self.log_tank[i][1]<releaseTime):
                    releaseTime=self.log_tank[i][1]
                    tank=self.TK[i][0]
        return tank

    def isSecureState(self):
        for time in self.time_ODF:
            if(time <self.time_ODT):
                self.SecureState = False
                return
        if self.schedule_pipe[-1][3]==self.schedule_pipe[-1][4]:
            self.SecureState = False
            return
        self.SecureState=True
    def close(self):
        print("close")