from pdb import set_trace
import torch
import triton
import triton.language as tl

def ceil_div(a,b):
    return int((a + b - 1)/b)
    
def test_pid_conds(conds, pid0=[0], pid1=[0], pid2=[0]):
    ''' Get the conds string and apply it to each pid'''

    conds = conds.replace(' ','').split(',')
    pids = [pid0[0], pid1[0], pid2[0]]
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond == '':
            continue
        op, threshold_val = cond[0], cond[1:]
        if op not in ["=",">","<","!"]:
            return ValueError("Check Condition")
        if op == "=":
            op = "=="
        elif op == "!":
            op = "!="
        if not eval(f'{pid}  {op} {threshold_val}'):
            return False
    return True

def breakpoint_if(conds, pid0=[0], pid1=[0], pid2=[0]):
    if(test_pid_conds(conds, pid0, pid1, pid2)):
        set_trace()


def print_if(val, conds, pid0=[0], pid1=[0], pid2=[0]):
    if(test_pid_conds(conds, pid0, pid1, pid2)):
        print(val)

