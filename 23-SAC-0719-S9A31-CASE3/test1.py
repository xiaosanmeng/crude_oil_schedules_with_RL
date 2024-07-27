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

list=pump_speed_combination_generator(1133.6,[840,1250,1370])
ans=(list[1]+list[2]*2+list[3]*3)*30
print(ans)