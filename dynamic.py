import numpy as np
import random

def q_transition(current_class, total_classes):
    # if np.random.binomial(size=1, n=1, p= 0.80)[0] == 0:
    #     action = int(np.random.normal(2, 10))
    #     return  (current_class + action)%total_classes
    # else:
    #     return current_class % total_classes

    # action = round(np.random.normal(0.2, 0.2))
    # action = round(np.random.normal(0.1, 0.30))
    # return (current_class + action) % total_classes


    # val = (int(current_class / task_size) + temp) % int(total_classes / task_size)
    action  = round(np.random.normal(0.03, 5.0))
    return current_class + action
    # return (val * 5 + random.randint(0, task_size)) % total_classes

current_class = 0
baseline = 0
total = 100000
max_class = -2
baseline2 = 0
for a in range(0,total):
    prev_class = current_class
    current_class = q_transition(current_class, 963)

    if current_class > 900:
        print(a)
        print("Accuracy = ", baseline/a)
        print("Accuracy 2 = ", baseline2 / a)
        quit()

    if prev_class == current_class:
        baseline+=1

    if current_class > max_class:
        max_class = current_class
    else:
        baseline2+=1


    print(current_class)

print("Baseline accuracy = ", baseline/total)
print("Accuracy 2 = ", baseline2 / total)