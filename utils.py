import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pulp import *


class Job:
    def __init__(self, inst, jid, Sj, rj, dj, wj):
        self.inst = inst  # instance

        self.id = jid  # job id
        self.S = Sj    # sequence of id of Tasks (list)
        self.r = rj    # release date
        self.d = dj    # due date
        self.w = wj    # weight

    def B(self):
        return self.inst.tasks[self.S[0]].B  # beginning date
    
    def C(self):
        return self.inst.tasks[self.S[-1]].C # completion date
    
    def cost(self):
        C = self.C()
        T = max(C - self.d, 0)
        U = 1 if T > 0 else 0
        return self.w * (C + self.inst.alpha*U + self.inst.beta*T)


class Task:
    def __init__(self, inst, tid, p, workers):
        self.inst = inst   # instance

        self.id = tid          # Task id
        self.p = p             # processing time
        self.workers = workers # possible Workers for this Task (list)

        self.B = None         # beginning date
        self.C = None         # completion date
        self.mid = None       # assigned machine id
        self.oid = None       # assigned operator id
        self.running = False  # is running
        self.done = False     # is completed


class Worker:
    def __init__(self, inst, mid, oid):
        self.inst = inst # instance

        self.mid = mid # machine id
        self.oid = oid # operator id


class Instance:
    def __init__(self, name):
        self.name = name

        self.J = None
        self.I = None
        self.M = None
        self.O = None
        self.alpha = None
        self.beta = None

        self.jobs = {}
        self.tasks = {}
        self.machines = {}  # machines availabilities
        self.operators = {} # operators availabilities

    def load(self, filename):
        with open(filename, 'rb') as f:
            inst = json.load(f)
        
        self.J = inst['parameters']['size']['nb_jobs']
        self.I = inst['parameters']['size']['nb_tasks']
        self.M = inst['parameters']['size']['nb_machines']
        self.O = inst['parameters']['size']['nb_operators']
        self.alpha = inst['parameters']['costs']['unit_penalty']
        self.beta = inst['parameters']['costs']['tardiness']

        for job in inst['jobs']:
            jid = job['job']
            self.jobs[jid] = self.parser_job(job)

        for task in inst['tasks']:
            tid = task['task']
            self.tasks[tid] = self.parser_task(task)

        # set all machines and operators as available
        for task in self.tasks:
            for worker in self.tasks[task].workers:
                self.machines[worker.mid] = True
                self.operators[worker.oid] = True

    def parser_job(self, job):
        jid = job['job']
        Sj  = job['sequence']
        rj  = job['release_date']
        dj  = job['due_date']
        wj  = job['weight']
        return Job(self, jid, Sj, rj, dj, wj)

    def parser_task(self, task):
        tid = task['task']
        p = task['processing_time']
        workers = []
        for machine in task['machines']:
            mid = machine['machine']
            for operator in machine['operators']:
                oid = operator
                workers.append(Worker(self, mid, oid))
        return Task(self, tid, p, workers)
    
    def greedy_solve(self):
        time = 0
        
        while (np.sum([t.done for t in self.tasks.values()]) < self.I):
            for t in self.tasks.values():
                if t.running:
                    if (time - t.B >= t.p):
                        t.running = False  # task ends
                        t.done = True      # task is done
                        self.machines[t.mid]  = True # free machine
                        self.operators[t.oid] = True # free operator
                        t.C = time  # set completion time

            for j in self.jobs.values():
                if time >= j.r: # it is past the release date of this job
                    for (t_idx, tid) in enumerate(j.S): # loop over tasks for this job
                        t = self.tasks[tid]      # get task from its id
                        for w in t.workers: # look for a worker (machine, operator) to execute task
                            if self.operators[w.oid] \
                            and self.machines[w.mid] \
                            and not t.running \
                            and not t.done \
                            and np.all([self.tasks[tid2].done for tid2 in j.S[:t_idx]]): # all job's previous tasks are done
                                t.running = True # set task to running
                                t.B = time       # set beginning time
                                t.mid = w.mid    # set machine id for task
                                t.oid = w.oid    # set operator id for task
                                self.machines[w.mid]  = False   # set machine to busy
                                self.operators[w.oid] = False   # set operator to busy

            time += 1 # time flows

    def cost(self):
        s = 0
        for job in self.jobs.values():
            C = job.C()
            T = max(C - job.d, 0)
            U = 1 if T > 0 else 0
            s += job.w * (C + self.alpha*U + self.beta*T)
        return s
    

class PuLP_Problem:
    def __init__(self, inst):
        """Instantiates a PuLP problem.

        Args:
            inst (Instance): the instance to solve
        """
        self.inst = inst
        self.prob = None
        self.solver = None

        self.B_vars = {}  # task beginnings vars
        self.C_vars = {}  # task completions vars
        self.T_vars = {}  # tardiness vars
        self.U_vars = {}  # unit penalty vars
        self.mach_assign = {}  # machine assignment vars
        self.op_assign = {}    # operator assignment vars

    def generate_problem(self):
        """Generates the PuLP problem."""

        print(f"Generating PuLP problem for {self.inst.name}...")

        self.prob = LpProblem(self.inst.name, LpMinimize)

        # big M
        M = 2*max([job.d for job in self.inst.jobs.values()])

        B_vars = {}
        C_vars = {}
        T_vars = {}
        U_vars = {}

        print(f"Adding jobs/tasks variables and constraints...")

        for job in self.inst.jobs.values():
            for tid in job.S:
                task = self.inst.tasks[tid]
                Bi = LpVariable(f"B{task.id}", cat=LpInteger)
                Ci = LpVariable(f"C{task.id}", cat=LpInteger)
                B_vars[tid] = Bi
                C_vars[tid] = Ci
                self.prob += Ci >= Bi + task.p  # C_i >= B_i + p_i
            
            self.prob += B_vars[job.S[0]] >= job.r  # B_i >= r_{j(i)}

            for idx in range(1, len(job.S)):
                self.prob += B_vars[job.S[idx]] >= C_vars[job.S[idx-1]]  # B_i >= C_{i-1}

            # tardiness
            Tj = LpVariable(f"T{job.id}", cat=LpInteger)
            T_vars[job.id] = Tj
            # T_j = max(0, C_j - d_j)
            self.prob += Tj >= 0
            self.prob += Tj >= C_vars[job.S[-1]] - job.d

            # unit penalty
            Uj = LpVariable(f"U{job.id}", cat=LpBinary)
            U_vars[job.id] = Uj
            # U_j = 1 if T_j > 0 else 0
            # M has to be greater than all tardinesses
            self.prob += M * Uj >= Tj

        print(f"Adding machines and operators variables and constraints...")

        mach_assign = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                mids = set([worker.mid for worker in task.workers])  # unique machine ids that can process this task
                for mid in mids:
                    mach_assign[tid, mid] = LpVariable(f"task{tid}_machine{mid}", cat=LpBinary)
                # each task is assigned to exactly one machine
                self.prob += lpSum([mach_assign[tid, mid] for mid in mids]) == 1

        op_assign = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                oids = set([worker.oid for worker in task.workers])  # unique operator ids that can process this task
                for oid in oids:
                    op_assign[tid, oid] = LpVariable(f"task{tid}_operator{oid}", cat=LpBinary)
                # each task is assigned to exactly one operator
                self.prob += lpSum([op_assign[tid, oid] for oid in oids]) == 1

        for mid in set([k[1] for k in mach_assign.keys()]):  # iterate over machines
            for (tid1, tid2) in combination(set([k[0] for k in mach_assign.keys() if k[1] == mid]), 2):
                indic1 = LpVariable(f"machine{mid}_C{tid1}>B{tid2}", cat=LpBinary)
                indic2 = LpVariable(f"machine{mid}_C{tid2}>B{tid1}", cat=LpBinary)
                # M has to be greater than complete running time of all jobs/tasks
                self.prob += C_vars[tid1] - B_vars[tid2] <= M * indic1  # C1 > B2 => indic1 = 1
                self.prob += C_vars[tid2] - B_vars[tid1] <= M * indic2  # C2 > B1 => indic2 = 1
                # if sum below is 4, then machine is simultaneously processing both tasks at some point
                self.prob += indic1 + indic2 + mach_assign[tid1, mid] + mach_assign[tid2, mid] <= 3

        for oid in set([k[1] for k in op_assign.keys()]):  # iterate over operators
            for (tid1, tid2) in combination(set([k[0] for k in op_assign.keys() if k[1] == oid]), 2):
                indic1 = LpVariable(f"operator{oid}_C{tid1}>B{tid2}", cat=LpBinary)
                indic2 = LpVariable(f"operator{oid}_C{tid2}>B{tid1}", cat=LpBinary)
                # M has to be greater than complete running time of all jobs/tasks
                self.prob += C_vars[tid1] - B_vars[tid2] <= M * indic1  # C1 > B2 => indic1 = 1
                self.prob += C_vars[tid2] - B_vars[tid1] <= M * indic2  # C2 > B1 => indic2 = 1
                # if sum below is 4, then operator is simultaneously handling both tasks at some point
                self.prob += indic1 + indic2 + op_assign[tid1, oid] + op_assign[tid2, oid] <= 3

        print(f"Adding objective function...")

        w = [job.w for job in self.inst.jobs.values()]  # job weights
        JC_vars = [C_vars[job.S[-1]] for job in self.inst.jobs.values()]  # job completion dates
        self.prob += lpSum([wj * (Cj + self.inst.alpha*Uj + self.inst.beta*Tj)
                       for wj, Cj, Uj, Tj in zip(w, JC_vars, U_vars.values(), T_vars.values())])
        
        # store variables
        self.B_vars, self.C_vars, self.T_vars, self.U_vars = B_vars, C_vars, T_vars, U_vars
        self.mach_assign, self.op_assign = mach_assign, op_assign

        print(f"PuLP problem generated.")

    def show_info(self):
        print(f"Problem {self.prob.name} has {self.prob.numVariables()} variables and {self.prob.numConstraints()} constraints")

    def warmup(self):
        """Sets intial values for variables to solve with warm start."""

        # set greedy beginning times
        for k, v in self.B_vars.items():
            v.setInitialValue(self.inst.tasks[k].B, check=True)
        # set greedy completion times
        for k, v in self.C_vars.items():
            v.setInitialValue(self.inst.tasks[k].C, check=True)
        # set greedy machine assignments
        for (tid, mid), v in self.mach_assign.items():
            if self.inst.tasks[tid].mid == mid:
                v.setInitialValue(1, check=True)
            else:
                v.setInitialValue(0, check=True)
        # set greedy operator assignments
        for (tid, oid), v in self.op_assign.items():
            if self.inst.tasks[tid].oid == oid:
                v.setInitialValue(1, check=True)
            else:
                v.setInitialValue(0, check=True)

    def set_solver(self, solver):
        """Sets solver for the problem."""

        self.solver = solver

    def solve(self):
        """Solves the PuLP problem."""
        
        self.prob.solve(self.solver)
        
    def show_status(self):
        print(f"Problem status: {LpStatus[self.prob.status]}\nObective value: {self.prob.objective.value()}")
        
    def savefile(self):
        """Saves the problem to disk as Mathematical Programming System file."""
        
        path = f"lp_problems/pulp_{self.inst.name}.mps"
        try:
            self.prob.writeMPS(path)
            print(f"Problem saved to {path}")
        except:
            print(f"Failed saving to file!")


class Gurobi_Problem:
    def __init__(self, inst):
        """Instantiates a PuLP problem.

        Args:
            inst (Instance): the instance to solve
        """
        self.inst = inst
        self.m = None

        self.B_vars = {}  # task beginnings vars
        self.C_vars = {}  # task completions vars
        self.T_vars = {}  # tardiness vars
        self.U_vars = {}  # unit penalty vars
        self.mach_assign = {}  # machine assignment vars
        self.op_assign = {}    # operator assignment vars

    def generate_problem(self):
        """Generates the Gurobi problem."""

        print(f"Generating Gurobi problem for {self.inst.name}...")

        self.m = gp.Model(self.inst.name)
        
        print(f"Greedy solving for time horizon estimation")
        self.inst.greedy_solve()
        T = int(max([j.C() for j in self.inst.jobs.values()]) * 1.25)

        B_vars = {}
        C_vars = {}
        T_vars = {}
        U_vars = {}
        
        print("Adding jobs/tasks variables and constraints...")

        for job in self.inst.jobs.values():
            for tid in job.S:
                task = self.inst.tasks[tid]
                Bi = self.m.addVar(name=f"B{task.id}", vtype=GRB.INTEGER)
                Ci = self.m.addVar(name=f"C{task.id}", vtype=GRB.INTEGER)
                B_vars[tid] = Bi
                C_vars[tid] = Ci
                self.m.addConstr(Ci >= Bi + task.p)  # C_i >= B_i + p_i

            self.m.addConstr(B_vars[job.S[0]] >= job.r)  # B_i >= r_{j(i)}

            for idx in range(1, len(job.S)):
                self.m.addConstr(B_vars[job.S[idx]] >= C_vars[job.S[idx-1]])  # B_i >= C_{i-1}

            # tardiness
            Tj = self.m.addVar(name=f"T{job.id}", vtype=GRB.INTEGER)
            T_vars[job.id] = Tj
            # T_j = max(0, C_j - d_j)
            # m.addConstr(Tj >= 0)  # redundant with integer default lower bound
            self.m.addConstr(Tj >= C_vars[job.S[-1]] - job.d)

            # unit penalty
            Uj = self.m.addVar(name=f"U{job.id}", vtype=GRB.BINARY)
            U_vars[job.id] = Uj
            # U_j = 1 if T_j > 0 else 0
            self.m.addConstr((Uj == 0) >> (Tj == 0))


        print("Creating running tables...")
        
        running_after_B  = self.m.addVars(range(1, T+1), range(1, self.inst.I+1), vtype=GRB.BINARY)
        running_before_C = self.m.addVars(range(1, T+1), range(1, self.inst.I+1), vtype=GRB.BINARY)
        running          = self.m.addVars(range(1, T+1), range(1, self.inst.I+1), vtype=GRB.BINARY)
        
        for t, tid in running_after_B.keys():
            self.m.addConstr((running_after_B[t, tid] == 0) >> (t <= B_vars[tid] - 1))
            self.m.addConstr((running_after_B[t, tid] == 1) >> (t >= B_vars[tid]))
        for t, tid in running_before_C.keys():
            self.m.addConstr((running_before_C[t, tid] == 0) >> (t >= C_vars[tid]))
            self.m.addConstr((running_before_C[t, tid] == 1) >> (t <= C_vars[tid] - 1))
        for t, tid in running.keys():
            self.m.addConstr(running[t, tid] == gp.and_(running_after_B[t, tid], running_before_C[t, tid]))

        
        print("Creating machines and operators task assignments tables...")
        
        mach_assign = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                mids = set([worker.mid for worker in task.workers])  # unique machine ids that can process this task
                for mid in mids:
                    mach_assign[tid, mid] = self.m.addVar(name=f"task_{tid}_machine_{mid}", vtype=GRB.BINARY)
                self.m.addConstr(sum([mach_assign[tid, mid] for mid in mids]) == 1)

        oper_assign = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                oids = set([worker.oid for worker in task.workers])  # unique operator ids that can process this task
                for oid in oids:
                    oper_assign[tid, oid] = self.m.addVar(name=f"task_{tid}_operator_{oid}", vtype=GRB.BINARY)
                self.m.addConstr(sum([oper_assign[tid, oid] for oid in oids]) == 1)
                
        print("Creating machines and operators business tables...")
        
        mach_business = self.m.addVars(range(1, T+1), range(1, self.inst.M+1), vtype=GRB.INTEGER)
        oper_business = self.m.addVars(range(1, T+1), range(1, self.inst.O+1), vtype=GRB.INTEGER)
        for t, mid in mach_business.keys():
            self.m.addConstr(mach_business[t, mid] <= 1)
        for t, oid in oper_business.keys():
            self.m.addConstr(oper_business[t, oid] <= 1)
            
        for t, mid in mach_business.keys():
            assigned_and_running = []
            for tid in [k[0] for k in mach_assign.keys() if k[1] == mid]:
                assigned_and_running.append(self.m.addVar(vtype=GRB.INTEGER))
                self.m.addConstr(assigned_and_running[-1] == gp.and_(mach_assign[tid, mid], running[t, tid]))
            self.m.addConstr(mach_business[t, mid] == sum(assigned_and_running))

        for t, oid in oper_business.keys():
            assigned_and_running = []
            for tid in [k[0] for k in oper_assign.keys() if k[1] == oid]:
                assigned_and_running.append(self.m.addVar(vtype=GRB.INTEGER))
                self.m.addConstr(assigned_and_running[-1] == gp.and_(oper_assign[tid, oid], running[t, tid]))
            self.m.addConstr(oper_business[t, oid] == sum(assigned_and_running))
        
        print("Adding objective function...")
        
        w = [job.w for job in self.inst.jobs.values()]  # job weights
        JC_vars = [C_vars[job.S[-1]] for job in self.inst.jobs.values()]  # job completion dates
        self.m.setObjective(sum([wj * (Cj + self.inst.alpha*Uj + self.inst.beta*Tj)
                            for wj, Cj, Uj, Tj in zip(w, JC_vars, U_vars.values(), T_vars.values())]),
                            GRB.MINIMIZE)
        
        # store variables
        self.B_vars, self.C_vars, self.T_vars, self.U_vars = B_vars, C_vars, T_vars, U_vars
        self.mach_assign, self.oper_assign = mach_assign, oper_assign

        print(f"Gurobi problem generated.")

    def warmup(self):
        """Sets intial values for variables to solve with warm start."""

        # set greedy beginning times
        for k, v in self.B_vars.items():
            v.Start = self.inst.tasks[k].B
        # set greedy completion times
        for k, v in self.C_vars.items():
            v.Start = self.inst.tasks[k].C
        # set greedy machine assignments
        for (tid, mid), v in self.mach_assign.items():
            if self.inst.tasks[tid].mid == mid:
                v.Start = 1
            else:
                v.Start = 0
        # set greedy operator assignments
        for (tid, oid), v in self.oper_assign.items():
            if self.inst.tasks[tid].oid == oid:
                v.Start = 1
            else:
                v.Start = 0

    def solve(self):
        """Solves the PuLP problem."""
        
        self.m.optimize()
        
    def show_status(self):
        sc = gurobipy.StatusConstClass
        status_codes = {sc.__dict__[k]: k for k, v in sc.__dict__.items() if isinstance(v, int)}
        print(f"Problem status: {status_codes[self.m.status]}\nObective value: {self.m.objVal}")
        
    def savefile(self):
        """Saves the problem to disk as Mathematical Programming System file."""
        
        path = f"lp_problems/gurobi_{self.inst.name}.mps"
        try:
            self.m.write(path)
            print(f"Problem saved to {path}")
        except:
            print(f"Failed saving to file!")
