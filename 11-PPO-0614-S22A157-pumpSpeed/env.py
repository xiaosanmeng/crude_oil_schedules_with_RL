

import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号











#（外部函数）泵速组合转换器:将泵速转为列表，1300=[0，0，0.6,0.4]
def pump_speed_combination_generator(rate):

    if (rate <= 833):
        x = rate / 833
        return [1 - x, x, 0, 0]
    if (rate <= 1250):
        x = (rate - 833) / 417
        return [0, 1 - x, x, 0]
    x = (rate - 1250) / 125
    return [0, 0, 1 - x, x]

class Env():

    def __init__(self):

        # TKi=[编号，类型，容量，存量]
        self.TK = [
            [1, 3, 34000, 27000],
            [2, 2, 34000, 30000],
            [3, 4, 34000, 27000],
            [4, 5, 34000, 30000],
            [5, 5, 34000, 25000],
            [6, 0, 34000, 0],
            [7, 0, 20000, 0],
            [8, 0, 20000, 0],
            [9, 0, 20000, 0]
        ]
        # RT=[编号，炼油类型，炼油量,炼油类型，炼油量]
        self.RT = [
            [1, 3, 27000, 1, 63000],
            [2, 2, 55200, 0, 0],
            [3, 4, 27000, 5, 55000, 6, 38000]
        ]
        #罐日志，数组下标+1=罐编号，元素为[chargeTIME,refinetime,[oilType,refineVolume]]
        self.log_tank = [[0, 0] for _ in range(len(self.TK))]
        #常量
        self.RESIDENCE_TIME = 6
        self.Charging_Tank_Switch_Overlap_Time=4
        self.F_SDU = [375, 230, 500]
        self.pipe_velocity=[0,833.3,1250,1375]
        #Mt and Mp 与上一算例一致
        self.Mt = [
              [0, 11, 12, 13, 10, 15],
              [11, 0, 11, 12, 13, 10],
              [12, 11, 0, 10, 12, 13],
              [13, 12, 10, 0, 11, 12],
              [10, 13, 12, 11, 0, 11],
              [15, 10, 13, 12, 11, 0]]
        self.Mp = [
              [0, 11, 12, 13, 7, 15],
              [10, 0, 9, 12, 13, 7],
              [13, 8, 0, 7, 12, 13],
              [13, 12, 7, 0, 11, 12],
              [7, 13, 12, 11, 0, 11],
              [15, 7, 13, 12, 11, 0]]
        # 是否为安全状态
        self.SecureState = True
        # 两个计划
        self.schedule_distiller = [[], [], []]  # [TANK,COT,V,START,END]
        self.schedule_pipe = []  # [TANK,COT,V,START,END,RATE]
        # 辅助time
        self.time_ODT = 0
        self.time_ODF = [0, 0, 0]
        #优化目标
        self.target=[0,0,0,0,0]
        self.preTarget=[0,0,0,5,0]#-----------------随算例改变而改变------------------------
        self.reward_weight=[1,1,1,1,0.2]

        self.refine()  # 预调度
        #管道残留原油类型(charge时需要同步修改，处默认情况下为0外，其他情况不允许为0
        self.residual_crude_oil_type_pipe=0
        #state
        self.INITSTATE = self.updataState()#INITSTATE需要在预调度之后
        self.n_states = len(self.INITSTATE)

        #action
        # action:(3*4*13)+1=156+1
        # 塔： 1.最急迫，  2.最不紧迫  3.选择同上一操作相同的蒸馏塔
        # 罐： 1.体积最小  2.体积最大  3.体积最接近 4.罐底混合最小
        # 泵速: 0 835（833）,880,925,970,1015,1060,1105,1150,1195,1240,1285,1330,1375=1+13

        #action的扩张，step函数不再以字典的方式读取：
            # # 创建字典
            # action_dict = {i: action for i, action in enumerate(self.actionsList)}
            # action = action_dict[action]
            #
            # # 操作定义
            # rate = self.pipe_velocity[(action % 10) - 1]
            # disitller = self.selectDistiller(action // 100) if rate != 0 else None
            # tank = self.selectTank((action % 100) // 10, disitller) if rate != 0 else None

        #新的动作解码方式:↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
        #

        # 传入action（0——144）：0=停运
        #                     1——144： 1-12=泵速833
        #                           : 13- 24=泵速880
        #                           : 25- 36=泵速925
        #                           :133-144=泵速1330
        #                           :145-156=泵速1375


        #泵速：[0 835（833）,880,925,970,1015,1060,1105,1150,1195,1240,1285,1330,1375]
        #rateListIndex=math.ceil(action/12)
        #   rateListIndex=0:0
        #   rateListIndex=1:833
        #   rateListIndex=2:880
        #   rateListIndex=13:1375


        #罐塔组合：[11,12,13,14,  21,22,23,24,  31,32,33,34]
        #TDListIndex=action%12>>TankAndDistiller
        #   TDListIndex=0:11
        #   TDListIndex=1:12
        #   TDListIndex=11:34


        #rate=self.rateList[rateListIndex]
        #disitller=TankAndDistiller//10
        #tank=TankAndDistiller%10

        self.rateList=[0,833,880,925,970,1015,1060,1105,1150,1195,1240,1285,1330,1375]#1+13
        self.TDList=[11,12,13,14,  21,22,23,24,  31,32,33,34]#12


        self.n_actions = (len(self.rateList)-1)*len(self.TDList)+1







    def reset(self):
        self.__init__()

    def step(self, action,render=False,rendertime=0.3):

        #return : state reward done

        #新的动作解码方式:↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
        rateListIndex=math.ceil(action/12)
        TDListIndex = action % 12
        TankAndDistiller=self.TDList[TDListIndex]

        rate=self.rateList[rateListIndex]

        disitller =self.selectDistiller( TankAndDistiller // 10) if rate != 0 else None

        tank=self.selectTank(TankAndDistiller%10,disitller) if rate != 0 else None

        #执行调度↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
        self.charge(disitller,tank,rate)
        self.refine()
        self.isSecureState()

        # ++++++++++++++++++++++++++++++++done++++++++++++++++++++++++++++++++++++++++++
        undistiller_volumn=0#未完成的原油体积
        for distiller in self.RT:
            # len=5>>>i=(2,4),len=7>>>i=(2,4,6)
            for i in range(2, len(distiller), 2):
                undistiller_volumn+=distiller[i]

        done = (undistiller_volumn== 0 or not self.SecureState)


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

            # ++++++++++++++++++++++++++++++++reward_energy++++++++++++++++++++++++++++++++++++++++++
            for schedule_pipe_i in self.schedule_pipe:
                rate = schedule_pipe_i[5]  # RATE取值=[0~1375]
                energy_time = (schedule_pipe_i[4] - schedule_pipe_i[3])  # 总时间
                rate_list = pump_speed_combination_generator(rate)  # 将泵速转为列表，1300=【0，0，0.6,0.4]

                self.target[4]+= (rate_list[1] + 2 * rate_list[2] + 3 * rate_list[3]) * energy_time
            self.target[4]=round(self.target[4],0)


            #++++++++++++++++++++++++++++++++++++REWARD
            reward=0
            for i in range(len(self.reward_weight)):
                reward += self.reward_weight[i]*(self.preTarget[i] - self.target[i])


            self.preTarget=self.target
            self.target=[0,0,0,0,0]

        else:
            reward=-400



        # ++++++++++++++++++++++++++++++++State计算++++++++++++++++++++++++++++++++++++++++++
        state=self.updataState()




        if render:
            self.render(rendertime)

        return state,reward,done

    def render(self,renderTime):
        #[TANK, COT, V, START, END]
        data = [self.schedule_distiller[0],
                self.schedule_distiller[1],
                self.schedule_distiller[2],
                self.schedule_pipe]
        color = {0:'w',
                 1:'#FFB6C1',
                 2:'#98FB98',
                 3:'r',
                 4:'y',
                 5:'c',
                 6:'#F4A460'}
        hatch = {0: '',
                 833.3: '/',
                 1250: '|',
                 1375: '-'
                 }
        # 画布设置，大小与分辨率
        plt.figure(figsize=(10, 4), dpi=100)
        plt.xlim(0,250)
        plt.ylim(0,5)
        plt.xticks(range(0, 241, 20))
        plt.yticks(range(0,6,1))

        #甘特图绘制
        for i in range(len(data)):
            for j in range(len(data[i])):
                plt.barh(y=4-i, width=(data[i][j][4]-data[i][j][3]),height=0.5, left=data[i][j][3], color=color[data[i][j][1]],edgecolor='k')
                plt.text(y=4-i,x=(data[i][j][4]+data[i][j][3])/2,s=("TK"+str(data[i][j][0]) if data[i][j][0]!=0 else ""),ha='center', va='center', color='k',fontsize=13)
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
        # 在原先的模型上进行该进，由i变量来控制sub_schedule
        for distiller in self.RT:
            #len=5>>>i=(2,4),len=7>>>i=(2,4,6)
            for i in range(2, len(distiller), 2):
                if(distiller[i]>0):
                    # 遍历罐列表
                    for tank in self.TK:
                        # 类型相同，存量不为0,并且可用
                        if (distiller[i] > 0 and tank[1] == distiller[i-1] and tank[3] != 0 and
                                self.log_tank[tank[0] - 1][0] <= self.time_ODF[distiller[0] - 1]):

                            oilType = tank[1]
                            refineVolume = tank[3]
                            if (len(self.schedule_distiller[distiller[0] - 1]) != 0):
                                startTime = self.schedule_distiller[distiller[0] - 1][-1][4]
                            else:
                                startTime = 0
                            endTime = round(startTime + (refineVolume / self.F_SDU[distiller[0] - 1]), 1)
                            schedule = [tank[0], oilType, refineVolume, startTime, endTime]
                            self.schedule_distiller[distiller[0] - 1].append(schedule)
                            self.time_ODF[distiller[0] - 1] = endTime

                            # ---------------------------------------------------------------
                            self.log_tank[tank[0] - 1].append([oilType, refineVolume])
                            self.log_tank[tank[0] - 1][1] = endTime+(0.5*self.Charging_Tank_Switch_Overlap_Time) # refine time更新
                            self.TK[tank[0] - 1][3] = 0
                            self.RT[distiller[0] - 1][i] -= refineVolume


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

            # 计算转运体积&类型，分两种情况：炼油类型1＆2&3
            sub_schedule_index=2
            while (sub_schedule_index< len(self.RT[distiller - 1]) and self.RT[distiller - 1][sub_schedule_index] == 0):
                sub_schedule_index+=2
            #入参distriller保证存在未完成的sub_schedule，因此可保证sub_schedule_index不越界

            volume = self.TK[tank - 1][2] if self.RT[distiller - 1][sub_schedule_index] > self.TK[tank - 1][
                2] else self.RT[distiller - 1][sub_schedule_index]
            cot = self.RT[distiller - 1][sub_schedule_index-1]

            #----------------------------------------origin--------------------------------------------------
            # if (self.RT[distiller - 1][2] != 0):  # 炼油任务1未完成
            #     volume = self.TK[tank - 1][2] if self.RT[distiller - 1][2] > self.TK[tank - 1][
            #         2] else self.RT[distiller - 1][2]
            #     cot = self.RT[distiller - 1][1]
            #
            # else:
            #     volume = self.TK[tank - 1][2] if self.RT[distiller - 1][4] > self.TK[tank - 1][
            #         2] else self.RT[distiller - 1][4]
            #     cot = self.RT[distiller - 1][3]
            #------------------------------------------------------------------------------------------------

            if (volume != 0):
                # 得出计划i
                schdule = [tank, cot, volume, startTime, startTime + volume / rate, rate]
                self.schedule_pipe.append(schdule)
                # 更新time_ODT
                self.time_ODT = round(startTime + volume / rate,1)
                # 充油
                self.TK[tank - 1][1] = cot
                self.TK[tank - 1][3] = volume
                self.log_tank[tank - 1][0] = round(startTime + volume / rate + self.RESIDENCE_TIME,1)+(0.5*self.Charging_Tank_Switch_Overlap_Time)
                self.residual_crude_oil_type_pipe=cot

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

        #parameter：type:1.最急迫，  2.最宽裕， 3. 管道混合成本最小（首次调度的情况默认选择蒸馏塔1) 4.居中（无居中情况时，采用最紧迫
        #return：   distiller的实际编号
        #》》》》》》》》》》》》》》》》》》》》1最紧迫，无边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》

        distiller = None
        if type==1:
            distiller=self.time_ODF.index(min(self.time_ODF))+1
        # 》》》》》》》》》》》》》》》》》》》》2最宽裕，存在边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        if type==2:
            max=0
            for i in range(len(self.time_ODF)):
                if(math.ceil(self.time_ODF[i])<240 and self.time_ODF[i]>max):
                    max=self.time_ODF[i]
                    distiller=i+1

        # 》》》》》》》》》》》》》》》》》》》》3管道混合成本最小》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        if type == 3:
            if self.residual_crude_oil_type_pipe==0:#（管道首次调度的情况 不包含停运操作，默认选择蒸馏塔1)
                distiller=1
            else:
                residual_crude_oil_type_pipe=self.residual_crude_oil_type_pipe
                min_mix_pipe_cost = float('inf')
                for RTi in self.RT:
                    if(math.ceil(self.time_ODF[RTi[0]-1])>=240):
                        continue
                    received_crude_oil_type_pipe=RTi[1] if RTi[2]!=0 else RTi[3]
                    mix_pipe_cost=self.Mp[residual_crude_oil_type_pipe-1][received_crude_oil_type_pipe-1]
                    if(min_mix_pipe_cost)>mix_pipe_cost:
                        min_mix_pipe_cost=mix_pipe_cost
                        distiller=RTi[0]

        #》》》》》》》》》》》》》》》》》》》》居中，存在边界问题》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
        # if type==3:
        #     count=0#非240 的元素,count==3的情况下才有居中情况
        #     for i in range(len(self.time_ODF)):
        #         if (math.ceil(self.time_ODF[i])<240):
        #             count+=1
        #     if(count==3):
        #         max=0
        #         second=0
        #         for i in range(len(self.time_ODF)):
        #             if(self.time_ODF[i]>max):
        #                 max=self.time_ODF[i]
        #
        #         for i in range(len(self.time_ODF)):
        #             if(self.time_ODF[i]<max and self.time_ODF[i]>second):
        #                 second=self.time_ODF[i]
        #                 distiller=i+1
        #     else:
        #         distiller=self.time_ODF.index(min(self.time_ODF))+1
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

        # ********************************4.罐底混合最小化*****************************************
        if type==4:
            #预接收原油的类型
            received_crude_oil_type_tank=self.RT[distiller-1][1] if self.RT[distiller-1][2]!=0 else self.RT[distiller-1][3]
            min_mix_tank_cost=float('inf')

            for emptyTank in emptyTK:

                if(len(self.log_tank[emptyTank[0]-1])>2):
                    residual_crude_oil_type_tank=self.log_tank[emptyTank[0]-1][-1][0]
                    mix_tank_cost=self.Mt[residual_crude_oil_type_tank-1][received_crude_oil_type_tank-1]
                else:
                    mix_tank_cost=0

                if(mix_tank_cost<min_mix_tank_cost):
                    min_mix_tank_cost=mix_tank_cost
                    tank=emptyTank[0]


        # # ********************************4.最近释放 *****************************************
        # if type==4:
        #     releaseTime=0
        #     for i in range(len(self.TK)):
        #         if(self.TK[i][3]==0 and self.log_tank[i][1]<=self.time_ODT and self.log_tank[i][1]>releaseTime):
        #             releaseTime=self.log_tank[i][1]
        #             tank=self.TK[i][0]
        #
        # # ********************************5.最早释放 *****************************************
        # if type == 5:
        #     releaseTime = 240
        #     for i in range(len(self.TK)):
        #         if(self.TK[i][3]==0 and self.log_tank[i][1]<=self.time_ODT and self.log_tank[i][1]<releaseTime):
        #             releaseTime=self.log_tank[i][1]
        #             tank=self.TK[i][0]


        return tank

    def isSecureState(self):
        stop_count=0
        for time in self.time_ODF:
            if(time <self.time_ODT):
                self.SecureState = False
                return
        for schedule in self.schedule_pipe:
            if(schedule[2]==0):
                stop_count+=1
            if(stop_count>5):
                self.SecureState = False
                return
        #self.SecureState=True

    def updataState(self):
        # 初始化状态22位
        # 0     :  [the completion rate of all refining tower]
        # 1·2 3 :  [the completion rate of each refining tower]
        # 4-13  :  [每个罐的存有情况%]
        # 14    :  [空罐数]
        # 15    :  [未使用过的空罐数]
        # 16    :  [曾使用过的空罐数]
        # 17    :  [所以罐的平均使用次数]
        # 18    :  [管道停运次数]
        # 19    :  [管道转运原油次数]
        # 20    :  [管道平均转运速率/1000]
        # 21    :  [管道混合次数]

        state = [0 for _ in range(22)]

        # 0     :  [the completion rate of all refining tower]
        undistiller_volumn = 0
        for rti in self.RT:
            undistiller_volumn += rti[2] + rti[4]
        state[0] = round(1 - (undistiller_volumn / (sum(self.F_SDU) * 240)), 2)
        # 1·2 3 :  [the completion rate of each refining tower]
        for i in range(3):
            state[i + 1] = round(1 - ((self.RT[i][2] + self.RT[i][4]) / (240 * self.F_SDU[i])), 2)
        # 4-13  :  [每个罐的存有情况%]
        for i in range(len(self.TK)):
            if (self.log_tank[i][1] <= (self.time_ODT) and self.TK[i][3] == 0):
                state[i + 4] = 0
            elif (self.TK[i][3] == 0):

                state[i + 4] = round(self.log_tank[i][-1][1] / self.TK[i][2], 2)
            else:
                state[i + 4] = round(self.TK[i][3] / self.TK[i][2], 2)

        # 14    :  [空罐数]
        # 15    :  [未使用过的空罐数]
        # 16    :  [曾使用过的空罐数]
        # 17    :  [mean所有罐的使用次数]
        for i in range(len(self.TK)):
            if (self.log_tank[i][1] <= (self.time_ODT) and self.TK[i][3] == 0):
                state[14] += 1
                if (len(self.log_tank[i]) == 2):
                    state[15] += 1
                else:
                    state[16] += 1
            state[17] += (len(self.log_tank[i]) - 2)
        state[17] /= len(self.TK)

        # [TANK,COT,V,START,END,(RATE]
        # 18    :  [管道停运次数]
        # 19    :  [管道转运原油次数]
        # 20    :  [管道平均转运速率/1000]
        # 21    :  [管道混合次数]
        if (len(self.schedule_pipe) != 0):
            for schedule in self.schedule_pipe:
                if (schedule[2] == 0):
                    state[18] += 1
                state[19] += 1
                state[20] += schedule[2] / 1000
            state[20] = round(state[20] / self.time_ODT, 2)

            cot_list = []
            for i in range(len(self.schedule_pipe)):
                if (self.schedule_pipe[i][1] != 0):
                    cot_list.append(self.schedule_pipe[i][1])
            for i in range(len(cot_list) - 1):
                if (cot_list[i + 1] != cot_list[i]):
                    state[21] += 1

        return state

    def close(self):
        print("close")