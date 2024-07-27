

import matplotlib.pyplot as plt
import math

import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号











def pump_speed_combination_generator(rate,pipe_velocity_list):
#（外部函数）泵速组合转换器:将泵速转为列表，1300=[0，0，0.6,0.4]

    if (rate <= pipe_velocity_list[0]):
        x = rate /  pipe_velocity_list[0]
        return [1 - x, x, 0, 0]
    if (rate <=  pipe_velocity_list[1]):
        x = (rate -  pipe_velocity_list[0]) / ( pipe_velocity_list[1]- pipe_velocity_list[0])
        return [0, 1 - x, x, 0]
    x = (rate -  pipe_velocity_list[1]) / ( pipe_velocity_list[2]- pipe_velocity_list[1])
    return [0, 0, 1 - x, x]

class Env():

    def __init__(self):

        self._max_episode_steps=30

        # TKi=[编号，类型，容量，存量]
        self.TK = [
            [1, 5, 16000, 8000],
            [2, 5, 34000, 30000],
            [3, 4, 34000, 30000],
            # [4, 0, 34000, 0],
            [5-1, 3, 34000, 30000],
            [6-1, 1, 16000, 16000],
            [7-1, 6, 20000, 16000],
            [8-1, 6, 16000, 5000]
            # [9, 0, 16000, 0],
            # [9, 0, 30000, 0]
        ]
        # RT=[编号，炼油类型，炼油量,炼油类型，炼油量]
        self.RT = [
            [1, 5, 38000, 1, 42000],
            [2, 6, 21000, 2, 49008],
            [3, 4, 30000, 3, 120000]
        ]
        #罐日志，数组下标+1=罐编号，元素为[chargeTIME,refinetime,[oilType,refineVolume]]
        self.log_tank = [[0, 0] for _ in range(len(self.TK))]
        #常量
        self.RESIDENCE_TIME = 6
        self.Charging_Tank_Switch_Overlap_Time=0
        self.F_SDU = [333.3, 291.7, 625]
        self.pipe_velocity_list=[840,1250,1370]
        #Mt and Mp 与上一算例一致
        self.Mt = [
              [0, 11, 12, 13, 10, 15],
              [11, 0, 11, 12, 13, 10],
              [12, 11, 0, 10, 12, 13],
              [13, 12, 10, 0, 11, 12],
              [10, 13, 12, 11, 0, 11],
              [15, 10, 13, 12, 11, 0]]
        self.Mt_mean=np.mean(np.array(self.Mt))

        self.Mp = [
              [0, 11, 12, 13, 7, 15],
              [10, 0, 9, 12, 13, 7],
              [13, 8, 0, 7, 12, 13],
              [13, 12, 7, 0, 11, 12],
              [7, 13, 12, 11, 0, 11],
              [15, 7, 13, 12, 11, 0]]
        self.Mp_mean=np.mean(np.array(self.Mp))
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
        self.preTarget=[0,0,7,7,0]#-----------------随算例改变而改变------------------------
        self.reward_weight=[1,1,1,1,1]

        self.refine()  # 预调度
        #管道残留原油类型(charge时需要同步修改，处默认情况下为0外，其他情况不允许为0
        self.residual_crude_oil_type_pipe=0
        #state
        self.INITSTATE = self.updataState()#INITSTATE需要在预调度之后
        self.n_states = len(self.INITSTATE)

        #action
        # action:(30)+1=31
        # 泵速: 停运
        #     1.840 2.1250  3.1370  4.dyna① 5.dyna②
        # 塔： 1.最急迫，    2.最不紧迫      3.管道混合成本最小
        # 罐： 1.优先体积最接近原则            2.优先罐底混合成本最小原则
        self.action_space=[0
                           ,111,112,121,122,131,132
                           ,211,212,221,222,231,232
                           ,311,312,321,322,331,332
                           ,411,412,421,422,431,432
                           ,511,512,521,522,531,532
                           ]
        self.n_actions = len(self.action_space)
        # self.action_space.sample = self.action_space[random.randint(0,self.n_actions-1)]







    def reset(self):
        self.__init__()

    def step(self, action,render=False,rendertime=0.25):

        #return : state reward done
        #param : 传入的参数a 为数组下标
        # print("preAction:",action,"curAction:",self.action_space[action])
        action = self.action_space[action]
        #新的动作解码方式:↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

        rateType=0
        distiller=None
        tank=None
        if action!=0:
            rateType = action//100
            distiller =self.selectDistiller( (action%100) // 10)
            tank=self.selectTank(action%10,distiller)
        #执行调度↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

        self.charge(distiller,tank,rateType)
        self.refine()
        self.isSecureState()

        #done
        undistiller_volumn=0#未完成的原油体积
        for distiller in self.RT:
            # len=5>>>i=(2,4),len=7>>>i=(2,4,6)
            for i in range(2, len(distiller), 2):
                undistiller_volumn+=distiller[i]

        done = (undistiller_volumn== 0 or not self.SecureState)




        if (self.SecureState):
        ################################################################################################
            # reward_mix_pipe
            cot_list = []
            for i in range(len(self.schedule_pipe)):
                if (self.schedule_pipe[i][1] != 0):
                    cot_list.append(self.schedule_pipe[i][1])
            for i in range(len(cot_list) - 1):
                pipe_cot_x = cot_list[i]
                pipe_cot_y = cot_list[i + 1]
                self.target[0] += self.Mp[pipe_cot_x - 1][pipe_cot_y - 1]
            # reward_mix_tank
            for log_tank_i in self.log_tank:
                for i in range(len(log_tank_i) - 3):
                    self.target[1] += self.Mt[log_tank_i[2 + i][0] - 1][log_tank_i[3 + i][0]  - 1]
            # reward_change_tank
            for schedule_distiller_i in self.schedule_distiller:
                self.target[2]+=len(schedule_distiller_i)
            # reward_used_tank
            for log_tanki in self.log_tank:
                if (len(log_tanki) > 2):
                    self.target[3]+= 1
            # reward_energy
            for schedule_pipe_i in self.schedule_pipe:
                rate = schedule_pipe_i[5]
                energy_time = (schedule_pipe_i[4] - schedule_pipe_i[3])
                rate_list = pump_speed_combination_generator(rate,self.pipe_velocity_list)

                self.target[4]+= (rate_list[1] + 2 * rate_list[2] + 3 * rate_list[3]) * energy_time
            self.target[4]=round(self.target[4],0)
        ################################################################################################


            #REWARD
            reward=0
            reward += self.reward_weight[0]*((self.preTarget[0]-self.target[0])/self.Mp_mean) if self.preTarget[0]!=self.target[0] else 0
            reward += self.reward_weight[1]*((self.preTarget[1]-self.target[1])/self.Mt_mean) if self.preTarget[1]!=self.target[1] else 0
            reward += self.reward_weight[2]*-1 if self.schedule_pipe[-1][-1]!=0 else 0
            reward += self.reward_weight[3]*((self.preTarget[3]-self.target[3]))
            reward += self.reward_weight[4]*((self.preTarget[4]-self.target[4])/(self.schedule_pipe[-1][4]-self.schedule_pipe[-1][3])) if self.schedule_pipe[-1][4]!=self.schedule_pipe[-1][3] else 0


            self.preTarget=self.target
            self.target=[0,0,0,0,0]

        else:
            reward=-100

        if (done and self.SecureState):
            self.time_ODT = 240
            # reward+=100

        # State
        state=self.updataState()

        if render and  (rendertime!=-1 or (done)):

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
        # hatch = {0: '',
        #          833.3: '/',
        #          1250: '|',
        #          1375: '-'
        #          }
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
                plt.text(y=4.35-i,x=(data[i][j][4]+data[i][j][3])/2,s=(str(data[i][j][-1]) if (i==3 and data[i][j][-1]!=0 )else ""),ha='center', va='center', color='k',fontsize=10)
        #其他设置
        #plt.grid(linestyle="--", alpha=0.5)
        # # XY轴标签
        plt.xlabel("调度时间/s")
        plt.ylabel("蒸馏塔/管道调度信息")

        if renderTime>0:
            plt.show(block=False)
            plt.pause(renderTime)
            plt.close()
        else:
            plt.show(block=True)
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


    def charge(self,distiller,tank,rateType):
        # parameter：蒸馏塔实际编号，充油罐实际编号，
        # 根据传入参数进行充油，更改env信息
        # tank=[编号，类型，容量，存量]  RT=[编号，炼油类型，炼油量,炼油类型，炼油量]
        #特殊判断
        if(tank==None or distiller==None):
            rateType=0
        # ==================计算开始时间=================================================
        if (len(self.schedule_pipe) != 0):  # 如果不为空，求出上一个管道运输信息[TANK,COT,V,START,END]
            startTime = self.schedule_pipe[-1][4]
        else:
            startTime = 0
        if (rateType != 0):

            # 计算转运体积&类型，分情况：炼油类型1＆2&3
            sub_schedule_index=2
            while (sub_schedule_index< len(self.RT[distiller - 1]) and self.RT[distiller - 1][sub_schedule_index] == 0):
                sub_schedule_index+=2
            #入参distriller保证存在未完成的sub_schedule，因此可保证sub_schedule_index不越界

            volume = self.TK[tank - 1][2] if self.RT[distiller - 1][sub_schedule_index] > self.TK[tank - 1][
                2] else self.RT[distiller - 1][sub_schedule_index]
            cot = self.RT[distiller - 1][sub_schedule_index-1]


            #待转运体积，其实volume不可能为nil
            if (volume != 0):

                rate=self.selectRate(rateType,volume,distiller,startTime)#← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← 在此处调用

                # 注意：如果由selectRate函数带回的rate值小于833.3的话，证明：该ODT实际上由（833.3+STOP）组成，即存在停运操作
                # 那么在charge函数中，应该生成两个ODT。再根据ODT来修改log_tank记录表。
                # 对与其他的泵速组合（eg:833.3+1250）,本质上仍是一个ODT，因此无需改动。


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
            #空罐统计
            emptyTK=[]
            for i in range(len(self.TK)):
                if (self.TK[i][3] == 0 and self.log_tank[i][1] <= self.time_ODT):
                    emptyTK.append(self.TK[i])

            #存在空罐，对应time_a
            if (len(emptyTK) != 0):
                mintank_v = min(tank[2] for tank in emptyTK)
                endTime_a = urge_time - mintank_v / 1250 - self.RESIDENCE_TIME
            #不存在空罐需要等待释放，对应time_b
            for i in range(len(self.TK)):
                if (self.TK[i][3] == 0 and self.log_tank[i][1] < endTime_b and self.log_tank[i][1] > startTime):  # 存量为0 and refineTime<240 and refineTime>startTime
                    endTime_b = self.log_tank[i][1]
            # time_a,time_b取最小值
            endTime = min(endTime_a, endTime_b)
            #边界
            if (endTime < startTime):
                endTime = startTime
            #schedule
            schdule = [0, 0, 0, startTime, endTime, 0]
            self.schedule_pipe.append(schdule)
            self.time_ODT = endTime


    def selectRate(self,type,volume,distiller,startTime):

        #param:
        #       type=[0,1,2,3,4,5]
        #       volume=int
        #       distiller=蒸馏塔的实际编号
        #       startTime=本次调度的开始时间
        #return:rate int
        #动态选择管道转运速率：
        #   1.管道最小速率为833.3。即管道转运速率的范围为（840-1370）case2，低于840的速率划分为两个ODT（833.3*k1+0*k2）的形式
        #   2.动态速率的选择考量：
        #
        #       计算前应得到本次ODT需要转运的体积，体积确定由distiller和tank求得
        #           ——应当改变rate,distiller tank的获取顺序
        #           ——转运体积的计算在原先charge函数内求得
        #               ——selectRate函数的调用由charge函数执行
        #
        #       type=1:当前状态下本次ODT所对应蒸馏塔中最晚的补给时间：latest_feed_time（大于当前时刻 同时要求满足带罐约束）
        #       type=2:当前状态下各个供油罐中最早的释放时间:earliest_release_time（大于当前时刻 同时要求满足带罐约束）
        #
        #
        #       通过确定本次ODT的结束时间来得出本次ODT采用的速率。
        #       ODT的结束时间：
        #           type=①latest_feed_time-self.RESIDENCE_TIME-self.Charging_Tank_Switch_Overlap_Time*α：
        #               在本次ODT完成后该供油罐经过滞留时间后供给给对应的distiller
        #               令endTime=latest_feed_time-self.RESIDENCE_TIME-self.Charging_Tank_Switch_Overlap_Time
        #
        #               if volume / (endTime - startTime) >= 840:
        #                   rate = volume / (endTime - startTime)
        #               else:
        #                   rate = 840
        #               此外:
        #                   如果volume/(endTime-startTime)>=1370，则系统已经处在不安全状态，调度失败
        #
        #           type=② earliest_release_time：nextState有更多的空罐可供选择:
        #
        #               volume/(earliest_release_time-startTime)>=840，<=1370:
        #                   rate=volume/(earliest_release_time-startTime)
        #
        #               若(earliest_release_time-startTime)/volume>=1370:
        #                   则舍弃当前的earliest_release_time，顺位取下一个
        #               上述条件无法满足的情况下默认为840
        #
        #
        #
        if type==0:
            return type
        if type>0 and type<4:
            return self.pipe_velocity_list[type-1]
        if type==4:

            #当前状态下本次ODT所对应蒸馏塔中最晚的补给时间：latest_feed_time（大于当前时刻 同时要求满足带罐约束）

            endTime=self.schedule_distiller[distiller-1][-1][4]-self.Charging_Tank_Switch_Overlap_Time*0.5-self.RESIDENCE_TIME

            #不安全状态
            if volume/(endTime-startTime)>self.pipe_velocity_list[-1]:
                #print("--------------------------The system is in an unsafe state-------------------------------------")
                return self.pipe_velocity_list[-1]
            #安全状态：
            if volume / (endTime - startTime) >= self.pipe_velocity_list[0]:
                rate = volume / (endTime - startTime)
            else:
                rate = self.pipe_velocity_list[0]

            return round(rate,1)
        if type==5:
            #当前状态下各个供油罐中最早的释放时间:earliest_release_time（大于当前时刻 同时要求满足带罐约束）
            min_release_time = startTime + volume / self.pipe_velocity_list[-1]  # 保证计算出来的rate不大于管道的最大流速
            max_release_time = startTime + volume / self.pipe_velocity_list[0]  # 保证计算出来的rate不小于管道的最小流速
            end_time=max_release_time    #本次调度的结束时间,有多个满足条件的情况下，选择最早的一个时间
            rate=self.pipe_velocity_list[1]
            for log_tank_i in self.log_tank:
                if( min_release_time<=log_tank_i[1] and max_release_time>=log_tank_i[1] and log_tank_i[1]<=end_time):
                    end_time = log_tank_i[1]
                    rate=volume/(end_time-startTime)

            return round(rate,1)







    def selectDistiller(self,type):

        #parameter：type:1.最急迫，  2.最不紧迫  3. 管道混合成本最小（首次调度的情况默认选择蒸馏塔1)
        #return：   distiller的实际编号


        distiller = None
        if type==1:
            distiller=self.time_ODF.index(min(self.time_ODF))+1

        if type==2:
            max=0
            for i in range(len(self.time_ODF)):
                if(math.ceil(self.time_ODF[i])<240 and self.time_ODF[i]>max):
                    max=self.time_ODF[i]
                    distiller=i+1

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


        #parameter:     type: 1.体积最接近 2.罐底混合成本最小
        #               distiller :蒸馏塔的实际编号
        #return:        tank的实际编号‘
        #********************************空罐判断*****************************************
        emptyTK=[]
        for tank in self.TK:
            if(tank[3]==0 and self.log_tank[tank[0]-1][1]<=self.time_ODT):
                emptyTK.append(tank)
        if len(emptyTK)==0:#此状态下没有空罐可供调度
            return None
        # 求出蒸馏塔需要的原油量，传入的蒸馏塔必定为未完成炼油任务的蒸馏塔，因此不必担心index越界
        index = 2
        while (self.RT[distiller - 1][index] == 0):
            index += 2
        needVolume = self.RT[distiller - 1][index]  # 蒸馏塔需要的原油量
        tank_received_type = self.RT[distiller - 1][index - 1]  # 油罐预接收的原油类型


        # ********************************1.体积最接近 *****************************************
        #原则①：优先选择充油罐容积能够满足蒸馏塔需要的炼油量的充油罐
        #原则②：在充油罐容积满足蒸馏塔需求的前提下，有多个充油罐的容积与蒸馏塔需求的原油量最接近时，选择罐底混合成本最小的油罐
        #原则③：罐底混合成本相同情况下，优先使用旧罐 （新油罐的罐底混合成本设置为0.01）
        if type==1:

            diff=[float("inf"),100,None,float("-inf"),100,None]#[diff>=0,罐底混合成本1,tank1编号,    diff<0，罐底混合成本2，tank2编号]
            # 左边为罐容量小于等于蒸馏塔需要的原油量，右边为罐容量大于蒸馏塔需要的原油量
            for emptyTank in emptyTK:
                tank_bottom_type=emptyTank[1]#残留罐底类型
                mix_cost=self.Mt[tank_bottom_type - 1][tank_received_type - 1] if tank_bottom_type!=0 else 0.01#罐底混合成本，新的油罐罐底混合成本设置为0.01
                if needVolume-emptyTank[2]>=0 and (needVolume-emptyTank[2]<diff[0] or (needVolume-emptyTank[2]==diff[0] and mix_cost<diff[1])):
                    diff[0]=needVolume-emptyTank[2]
                    diff[1]=mix_cost
                    diff[2]=emptyTank[0]
                if needVolume-emptyTank[2]<0 and (needVolume-emptyTank[2]>diff[3] or (needVolume-emptyTank[2]==diff[3] and mix_cost<diff[4])):
                    diff[3]=needVolume-emptyTank[2]
                    diff[4]=mix_cost
                    diff[5]=emptyTank[0]
            return diff[5] if diff[5]!=None else diff[2]


        # ********************************2.罐底混合成本最小化*****************************************
        #原则①新油罐的罐底混合成本为0.01，其目的便是：在罐底混合成本都为0时，优先使用旧的油罐
        #原则②：罐底混合成本相同，则选择充油罐容积能够满足蒸馏塔需要的炼油量的充油罐
        #原则③：在罐底混合成本相同的条件下，存在多个充油罐容积满足蒸馏塔需求时，选择充油罐容积与蒸馏塔需要的炼油量最接近的充油罐
        if type==2:

            diff=[float('inf'),None]#罐底混合成本，tank的信息【编号，罐底，容积，存量】

            for emptyTank in emptyTK:
                tank_bottom_type = emptyTank[1]  # 残留罐底类型
                mix_cost = self.Mt[tank_bottom_type - 1][tank_received_type - 1] if tank_bottom_type != 0 else 0.01  # 罐底混合成本

                if mix_cost<diff[0] :
                    diff[0]=mix_cost
                    diff[1]=emptyTank
                if mix_cost==diff[0]:
                    if emptyTank[2]>=needVolume and emptyTank[2]<diff[1][2]:
                        diff[1]=emptyTank
                    if emptyTank[2]<=needVolume and emptyTank[2]>diff[1][2]:
                        diff[1]=emptyTank
            return diff[1][0]

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
        # 初始化状态9位
        # 0 1 2 :  [the completion rate of each refining tower]
        # 3 4 5 ： [是否存在罐底残留原油的类型与炼油厂所需类型原油相同类型原油的供油罐 0 or 1]
        # 6 7 8 ： [每个蒸馏塔的最晚派遣供油罐时间-当前时间]/240
        #
        # ********************************空罐队列*****************************************
        emptyTankList = []
        for tank in self.TK:
            if (tank[3] == 0 and self.log_tank[tank[0] - 1][1] <= self.time_ODT):
                emptyTankList.append(tank)


        state = [0 for _ in range(9)]

        # `012:[the completion rate of each refining tower]
        for i in range(len(self.RT)):
            unrefine_volume = 0
            for index in range(2, len(self.RT[i]), 2):
                unrefine_volume += self.RT[i][index]
            state[i] = round(1 - (unrefine_volume / (240 * self.F_SDU[i])), 2)

        # 345:[是否存在罐底残留原油的类型与炼油厂所需类型原油相同类型原油的供油罐 0 or 1]
        if len(emptyTankList)!=0:
            for i in range(len(self.RT)):
                if self.time_ODF[i]<240:
                    index=2
                    while(self.RT[i][index]==0 ):
                        index+=2
                    required_COT=self.RT[i][index-1]
                    for emptyTank in emptyTankList:
                        if(emptyTank[1]==required_COT):
                            state[i+3]=1
                            break
                else:
                    state[i+3]=0
        # 7 8 9 ： [每个蒸馏塔的最晚派遣供油罐时间-当前时间]/240
        for i in range(len(self.RT)):
            latestFeedTime=self.schedule_distiller[i][-1][4]-self.Charging_Tank_Switch_Overlap_Time*0.5-self.RESIDENCE_TIME
            state[i+6]=(latestFeedTime-self.time_ODT)/240 if self.schedule_distiller[i][-1][4]!=240 else 0


        return state

    def close(self):
        print("close")