import numpy as np
import json

class Job:
    def __init__(self, jid, Sj, rj, dj, wj):
        self.id = jid # job id
        self.S = Sj   # sequence of id of Tasks (list)
        self.r = rj   # release date
        self.d = dj   # due date
        self.w = wj   # weight
    
    def __str__(self):
        return f"id={self.id:>3}  w={self.w:>2}  r:{self.r:>3}  d:{self.d:>3}  " + \
               f"min_duration={self.min_duration():>3}  B:{self.B():>3}  " + \
               f"C:{self.C():>3}  late?{self.C()>self.d:>1}  delay={max(0, self.C()-self.d):>2}  cost={self.cost():>4}"
    
    def min_duration(self):
        return np.sum([tasks[tid].p for tid in self.S])
    
    def min_remaining_duration(self):
        return np.sum([tasks[tid].p for tid in self.S if not tasks[tid].done])
    
    def B(self):
        return tasks[self.S[0]].B # beginning date
    
    def C(self):
        return tasks[self.S[-1]].C # completion date
    
    def cost(self):
        C = self.C()
        T = max(C - self.d, 0)
        U = 1 if T > 0 else 0
        return self.w * (C + alpha*U + beta*T)

class Task:
    def __init__(self, tid, jid, pi, workers):
        self.id = tid          # Task id
        self.jid = jid         # Job id of corresponding Job
        self.p = pi            # processing time
        self.workers = workers # possible Workers for this Task (list)
        self.B = None          # beginning date
        self.C = None          # completion date
        self.running = False
        self.done = False
        self.mid = None        # id of Machine on which task runs
        self.oid = None        # id of Operator doing this task
        
    def __str__(self):
        j = jobs[self.jid]
        delay = int(self.B - (j.r + np.sum([tasks[tid2].p for tid2 in jobs[self.jid].S if tid2 < self.id])))
        return f"id={self.id:>3}  job={self.jid:>3}  begin:{self.B:>3}  complete:{self.C:>3}  " + \
               f"(time:{self.p:>2})  (delay:{delay:>2})  " + \
               f"machine:{self.mid:>2}  operator:{self.oid:>2}"
        
class Worker:
    def __init__(self, mid, oid):
        self.mid = mid # id of Machine used
        self.oid = oid # id of Operator working
        
    def __str__(self):
        return f"machine={self.mid} operator:{self.oid}"
    
class WorkUnit:
    def __init__(self, time, jid, t_idx, tid, w):
        self.time = time   # time at which WorkUnit is considered to start
        self.jid = jid     # Job id
        self.t_idx = t_idx # index of Task in Job's Tasks sequence
        self.tid = tid     # Task id
        self.w = w         # Worker
    
def parser_job(job):
    jid = job['job']
    Sj  = job['sequence']
    rj  = job['release_date']
    dj  = job['due_date']
    wj  = job['weight']
    return Job(jid, Sj, rj, dj, wj)

def parser_task(task, jid):
    tid = task['task']
    pi  = task['processing_time']
    workers = []
    for machine in task['machines']:
        mid = machine['machine']
        for operator in machine['operators']:
            oid = operator
            workers.append(Worker(mid, oid))
    return Task(tid, jid, pi, workers)

def parser_inst(inst):
    J = inst['parameters']['size']['nb_jobs']
    I = inst['parameters']['size']['nb_tasks']
    M = inst['parameters']['size']['nb_machines']
    O = inst['parameters']['size']['nb_operators']
    alpha = inst['parameters']['costs']['unit_penalty']
    beta = inst['parameters']['costs']['tardiness']
    
    machines  = {} # storing machines' availability by id
    operators = {} # storing operators' availability by id
    jobs  = (J+1) * [None]                          # list of Jobs
    jobs[0] = Job(jid=0, Sj=[], rj=-1, dj=-1, wj=0) # dummy element 0
    tasks = (I+1) * [None]                          # list of Tasks
    tasks[0] = Task(tid=0, jid=0, pi=0, workers=[]) # dummy element 0
    for job in inst['jobs']:
        jid = job['job']
        jobs[jid] = parser_job(job) # add Job in list of Jobs
        for task in inst['tasks']:
            tid = task['task']
            if tid in jobs[jid].S:
                tasks[tid] = parser_task(task, jid) # add task in list of Tasks
                for machine in task['machines']:
                    mid = machine['machine']
                    for operator in machine['operators']:
                        oid = operator
                        machines[mid]  = True # set machine to available
                        operators[oid] = True # set operator to available
    
    return (J, I, M, O, alpha, beta, jobs, tasks, machines, operators)

def optim2():
    ########## SUBROUTINE ##########
    def get_admissible_work_units():
        work_units = []
        for j in jobs:
            if time >= j.r: # it is past the release date of this Job
                for (t_idx, tid) in enumerate(j.S): # loop over Tasks for this Job
                    t = tasks[tid] # get Task from its id
                    if (not t.running) \
                    and (not t.done) \
                    and np.all([tasks[tid2].done for tid2 in j.S[:t_idx]]): # all Job's previous Tasks are done
                        for w in t.workers: # look for a Worker (machine, operator) to execute Task
                            if operators[w.oid] and machines[w.mid]: # Worker is free
                                work_units.append(WorkUnit(time, j.id, t_idx, tid, w))
        return work_units
    ################################
    
    time = 0
    while (np.sum([t.done for t in tasks]) < I):
        
        ################# UPDATE RUNNING TASKS ##################
        for t in tasks:
            if t.running:
                if (time - t.B >= t.p):
                    t.running = False # task ends
                    t.done = True     # task is done
                    machines[t.mid]  = True # free machine
                    operators[t.oid] = True # free operator
                    t.C = time              # set completion time
        #########################################################
        
        work_units = get_admissible_work_units()
        while len(work_units) > 0: # while we can still start tasks
            work_units = sorted(work_units, key=WU_weight)
            chosen_work_unit = work_units[-1] # choose the work unit with the biggest weight to finish it at the earliest
            t = tasks[chosen_work_unit.tid]
            w = chosen_work_unit.w
            t.running = True # set task to running
            t.B = time       # set beginning time
            t.mid = w.mid    # set machine id for task
            t.oid = w.oid    # set operator id for task
            machines[w.mid]  = False   # set machine to busy
            operators[w.oid] = False   # set operator to busy
            
            work_units = get_admissible_work_units()
                    
        time += 1 # time flows
        
def WU_weight(work_unit):
    j = jobs[work_unit.jid]  # corresponding Job
    t = tasks[work_unit.tid] # corresponding Task
    C = work_unit.time + j.min_remaining_duration() # proxy for job completion time
    T = max(C - j.d, 0)
    U = 1 if T > 0 else 0
    available_machines = {}
    available_operators = {}
    for worker in t.workers:
        available_machines[worker.mid] = True
        available_operators[worker.oid] = True
    #return w1*j.w*C + w2*j.w*j.d + w3*j.w*alpha*U + w4*j.w*beta*T + w5*len(t.workers)
    return w1*j.w*C + w2*j.w*j.d + w3*j.w*len(available_machines) + w4*j.w*len(available_operators)
        
def cost(jobs):
    s = 0
    for job in jobs[1:]:
        C = job.C()
        T = max(C - job.d, 0)
        U = 1 if T > 0 else 0
        s += job.w * (C + alpha*U + beta*T)
    return s

if __name__=='__main__':
    in_filename = "instances/large.json"
    with open(in_filename, 'rb') as f:
        inst = json.load(f)
    (J, I, M, O, alpha, beta, jobs, tasks, machines, operators) = parser_inst(inst)
    print(f"J={J}\nI={I}\nM={M}\nO={O}\nα={alpha}\nβ={beta}\n")
    best_cost = 10**6
    best_weights = (None, None, None, None, None)
    for _ in range(1, 200+1):
        if _ % 10 == 0:
            print(f"{_}... ", end=' ')
            print(best_cost, best_weights)
        w1 = 1
        #w2 = np.random.normal()
        #w2 = np.random.uniform(-0.9, -0.7)
        #w2 = np.random.uniform(-1.2, -0.8)
        w2 = -1
        w3 = -np.random.random()
        w4 = -np.random.random()
        w5 = np.random.normal()
        (J, I, M, O, alpha, beta, jobs, tasks, machines, operators) = parser_inst(inst)
        optim2()
        candidate_cost = cost(jobs)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_weights = (w1, w2, w3, w4, w5)
