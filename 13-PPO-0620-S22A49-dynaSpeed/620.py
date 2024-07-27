import random


def pump_speed_combination_generator(rate):

    if (rate <= 833):
        x = rate / 833
        return [1 - x, x, 0, 0]
    if (rate <= 1250):
        x = (rate - 833) / 417
        return [0, 1 - x, x, 0]
    x = (rate - 1250) / 125
    return [0, 0, 1 - x, x]

for i in range(10):
    x=random.randint(800,1375)
    print(x,"-",pump_speed_combination_generator(x))
    print("")