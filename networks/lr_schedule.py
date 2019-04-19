
def half_lr(init_lr, ep):
    lr = init_lr / 2**ep

    return lr

def step_lr(ep):
    if ep < 20:
        lr = 0.01
    elif ep < 50:
        lr = 0.001
    elif ep < 100:
        lr = 0.0001
    elif ep < 150:
        lr = 0.00001
    elif ep < 200:
        lr = 0.000001
    else:
        ep = 0.0000001
    return lr
