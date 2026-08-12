from __future__ import annotations

import json
from itertools import combinations
from typing import TypedDict, cast

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from pulp import (
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpSolver,
    LpStatus,
    LpVariable,
    lpSum,
)
from pydantic import BaseModel, ConfigDict


class JobData(TypedDict):
    job: int
    sequence: list[int]
    release_date: int
    due_date: int
    weight: int


class MachineData(TypedDict):
    machine: int
    operators: list[int]


class TaskData(TypedDict):
    task: int
    processing_time: int
    machines: list[MachineData]


class InstanceSizeData(TypedDict):
    nb_jobs: int
    nb_tasks: int
    nb_machines: int
    nb_operators: int


class InstanceParametersData(TypedDict):
    size: InstanceSizeData
    costs: dict[str, int]


class InstanceData(TypedDict):
    parameters: InstanceParametersData
    jobs: list[JobData]
    tasks: list[TaskData]


class Job(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    inst: Instance  # instance
    id: int  # job id
    S: list[int]  # sequence of id of Tasks (list)
    r: int  # release date
    d: int  # due date
    w: int  # weight
    T: list[int]  # list of tasks for this job

    def B(self) -> int:
        beginning = self.inst.tasks[self.S[0]].B
        assert beginning is not None
        return beginning  # beginning date

    def C(self) -> int:
        completion = self.inst.tasks[self.S[-1]].C
        assert completion is not None
        return completion  # completion date

    def cost(self) -> int:
        C = self.C()
        T = max(C - self.d, 0)
        U = 1 if T > 0 else 0
        assert self.inst.alpha is not None
        assert self.inst.beta is not None
        alpha = self.inst.alpha
        beta = self.inst.beta
        return self.w * (C + alpha * U + beta * T)


class Task(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int  # Task id
    p: int  # processing time
    workers: list[Worker]  # possible Workers for this Task (list)
    j: int  # this Task belongs to
    B: int | None = None  # beginning date
    C: int | None = None  # completion date
    mid: int | None = None  # assigned machine id
    oid: int | None = None  # assigned operator id
    running: bool = False  # is running
    done: bool = False  # is completed


class Worker(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    mid: int  # id of Machine used
    oid: int  # id of Operator working


class Instance:
    def __init__(self, name: str) -> None:
        self.name = name

        self.J: int | None = None
        self.I: int | None = None
        self.M: int | None = None
        self.O: int | None = None
        self.alpha: int | None = None
        self.beta: int | None = None

        self.jobs: dict[int, Job] = {}
        self.tasks: dict[int, Task] = {}
        self.machines: dict[int, bool] = {}  # machines availabilities
        self.operators: dict[int, bool] = {}  # operators availabilities
        self.task2job: dict[int, int] = {}  # task to job mapping

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            inst: InstanceData = json.load(f)

        self.J = inst["parameters"]["size"]["nb_jobs"]
        self.I = inst["parameters"]["size"]["nb_tasks"]
        self.M = inst["parameters"]["size"]["nb_machines"]
        self.O = inst["parameters"]["size"]["nb_operators"]
        self.alpha = inst["parameters"]["costs"]["unit_penalty"]
        self.beta = inst["parameters"]["costs"]["tardiness"]

        for job in inst["jobs"]:
            jid = job["job"]
            self.jobs[jid] = self.parser_job(job)

        for task in inst["tasks"]:
            tid = task["task"]
            self.tasks[tid] = self.parser_task(task)

        # set all machines and operators as available
        for tid in self.tasks:
            for worker in self.tasks[tid].workers:
                self.machines[worker.mid] = True
                self.operators[worker.oid] = True

    def parser_job(self, job: JobData) -> Job:
        jid = job["job"]
        Sj = job["sequence"]
        rj = job["release_date"]
        dj = job["due_date"]
        wj = job["weight"]
        tj = []
        for tid in Sj:
            self.task2job[tid] = jid
            tj.append(tid)
        return Job(inst=self, id=jid, S=Sj, r=rj, d=dj, w=wj, T=tj)

    def parser_task(self, task: TaskData) -> Task:
        tid = task["task"]
        p = task["processing_time"]
        j = self.task2job[tid]
        workers = []
        for machine in task["machines"]:
            mid = machine["machine"]
            for operator in machine["operators"]:
                oid = operator
                workers.append(Worker(mid=mid, oid=oid))
        return Task(id=tid, p=p, workers=workers, j=j)

    def greedy_solve(self) -> None:
        time = 0

        while np.sum([t.done for t in self.tasks.values()]) < self.I:
            for t in self.tasks.values():
                if t.running:
                    assert t.B is not None
                    assert t.mid is not None
                    assert t.oid is not None
                    if time - t.B >= t.p:
                        t.running = False  # task ends
                        t.done = True  # task is done
                        self.machines[t.mid] = True  # free machine
                        self.operators[t.oid] = True  # free operator
                        t.C = time  # set completion time

            for j in self.jobs.values():
                if time >= j.r:  # it is past the release date of this job
                    for t_idx, tid in enumerate(j.S):  # loop over tasks for this job
                        t = self.tasks[tid]  # get task from its id
                        for w in (
                            t.workers
                        ):  # look for a worker (machine, operator) to execute task
                            if (
                                self.operators[w.oid]
                                and self.machines[w.mid]
                                and not t.running
                                and not t.done
                                and np.all(
                                    [self.tasks[tid2].done for tid2 in j.S[:t_idx]]
                                )
                            ):  # all job's previous tasks are done
                                t.running = True  # set task to running
                                t.B = time  # set beginning time
                                t.mid = w.mid  # set machine id for task
                                t.oid = w.oid  # set operator id for task
                                self.machines[w.mid] = False  # set machine to busy
                                self.operators[w.oid] = False  # set operator to busy

            time += 1  # time flows

    def cost(self) -> int:
        s = 0
        for job in self.jobs.values():
            C = job.C()
            T = max(C - job.d, 0)
            U = 1 if T > 0 else 0
            assert self.alpha is not None
            assert self.beta is not None
            s += job.w * (C + self.alpha * U + self.beta * T)
        return s


class PuLP_Problem:
    def __init__(self, inst: Instance) -> None:
        """Instantiates a PuLP problem.

        Args:
            inst (Instance): the instance to solve
        """
        self.inst = inst
        self.prob: LpProblem | None = None
        self.solver: LpSolver | None = None

        self.B_vars: dict[int, LpVariable] = {}  # task beginnings vars
        self.C_vars: dict[int, LpVariable] = {}  # task completions vars
        self.T_vars: dict[int, LpVariable] = {}  # tardiness vars
        self.U_vars: dict[int, LpVariable] = {}  # unit penalty vars
        self.mach_assign: dict[
            tuple[int, int], LpVariable
        ] = {}  # machine assignment vars
        self.op_assign: dict[
            tuple[int, int], LpVariable
        ] = {}  # operator assignment vars

    def generate_problem(self) -> None:
        """Generates the PuLP problem."""

        print(f"Generating PuLP problem for {self.inst.name}...")

        self.prob = LpProblem(self.inst.name, LpMinimize)

        # big M
        M = 2 * max([job.d for job in self.inst.jobs.values()])

        B_vars: dict[int, LpVariable] = {}
        C_vars: dict[int, LpVariable] = {}
        T_vars: dict[int, LpVariable] = {}
        U_vars: dict[int, LpVariable] = {}

        print("Adding jobs/tasks variables and constraints...")

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
                self.prob += (
                    B_vars[job.S[idx]] >= C_vars[job.S[idx - 1]]
                )  # B_i >= C_{i-1}

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

        print("Adding machines and operators variables and constraints...")

        mach_assign: dict[tuple[int, int], LpVariable] = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                mids = {worker.mid for worker in task.workers}
                for mid in mids:
                    mach_assign[tid, mid] = LpVariable(
                        f"task{tid}_machine{mid}", cat=LpBinary
                    )
                # each task is assigned to exactly one machine
                self.prob += lpSum([mach_assign[tid, mid] for mid in mids]) == 1

        op_assign: dict[tuple[int, int], LpVariable] = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                oids = {worker.oid for worker in task.workers}
                for oid in oids:
                    op_assign[tid, oid] = LpVariable(
                        f"task{tid}_operator{oid}", cat=LpBinary
                    )
                # each task is assigned to exactly one operator
                self.prob += lpSum([op_assign[tid, oid] for oid in oids]) == 1

        for mid in {k[1] for k in mach_assign}:  # iterate over machines
            for tid1, tid2 in combinations(
                {k[0] for k in mach_assign if k[1] == mid}, 2
            ):
                indic1 = LpVariable(f"machine{mid}_C{tid1}>B{tid2}", cat=LpBinary)
                indic2 = LpVariable(f"machine{mid}_C{tid2}>B{tid1}", cat=LpBinary)
                # M has to be greater than complete running time of all jobs/tasks
                self.prob += (
                    C_vars[tid1] - B_vars[tid2] <= M * indic1
                )  # C1 > B2 => indic1 = 1
                self.prob += (
                    C_vars[tid2] - B_vars[tid1] <= M * indic2
                )  # C2 > B1 => indic2 = 1
                # if sum below is 4, then machine is simultaneously processing both tasks at some point
                self.prob += (
                    indic1 + indic2 + mach_assign[tid1, mid] + mach_assign[tid2, mid]
                    <= 3
                )

        for oid in {k[1] for k in op_assign}:  # iterate over operators
            for tid1, tid2 in combinations({k[0] for k in op_assign if k[1] == oid}, 2):
                indic1 = LpVariable(f"operator{oid}_C{tid1}>B{tid2}", cat=LpBinary)
                indic2 = LpVariable(f"operator{oid}_C{tid2}>B{tid1}", cat=LpBinary)
                # M has to be greater than complete running time of all jobs/tasks
                self.prob += (
                    C_vars[tid1] - B_vars[tid2] <= M * indic1
                )  # C1 > B2 => indic1 = 1
                self.prob += (
                    C_vars[tid2] - B_vars[tid1] <= M * indic2
                )  # C2 > B1 => indic2 = 1
                # if sum below is 4, then operator is simultaneously handling both tasks at some point
                self.prob += (
                    indic1 + indic2 + op_assign[tid1, oid] + op_assign[tid2, oid] <= 3
                )

        print("Adding objective function...")

        w = [job.w for job in self.inst.jobs.values()]  # job weights
        JC_vars = [
            C_vars[job.S[-1]] for job in self.inst.jobs.values()
        ]  # job completion dates
        self.prob += lpSum(
            [
                wj * (Cj + self.inst.alpha * Uj + self.inst.beta * Tj)
                for wj, Cj, Uj, Tj in zip(w, JC_vars, U_vars.values(), T_vars.values())
            ]
        )

        # store variables
        self.B_vars, self.C_vars, self.T_vars, self.U_vars = (
            B_vars,
            C_vars,
            T_vars,
            U_vars,
        )
        self.mach_assign, self.op_assign = mach_assign, op_assign

        print("PuLP problem generated.")

    def show_info(self) -> None:
        assert self.prob is not None
        print(
            f"Problem {self.prob.name} has {self.prob.numVariables()} variables and {self.prob.numConstraints()} constraints"
        )

    def warmup(self) -> None:
        """Sets intial values for variables to solve with warm start."""

        assert self.prob is not None

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

    def set_solver(self, solver: LpSolver) -> None:
        """Sets solver for the problem."""

        self.solver = solver

    def solve(self) -> None:
        """Solves the PuLP problem."""

        assert self.prob is not None
        self.prob.solve(self.solver)

    def show_status(self) -> None:
        assert self.prob is not None
        print(
            f"Problem status: {LpStatus[self.prob.status]}\nObective value: {self.prob.objective.value()}"
        )

    def savefile(self) -> None:
        """Saves the problem to disk as Mathematical Programming System file."""

        assert self.prob is not None
        path = f"lp_problems/pulp_{self.inst.name}.mps"
        try:
            self.prob.writeMPS(path)
            print(f"Problem saved to {path}")
        except (OSError, ValueError, RuntimeError):
            print("Failed saving to file!")


class Gurobi_Problem:
    def __init__(self, inst: Instance) -> None:
        """Instantiates a PuLP problem.

        Args:
            inst (Instance): the instance to solve
        """
        self.inst = inst
        self.m: gp.Model | None = None

        self.B_vars: dict[int, gp.Var] = {}  # task beginnings vars
        self.C_vars: dict[int, gp.Var] = {}  # task completions vars
        self.T_vars: dict[int, gp.Var] = {}  # tardiness vars
        self.U_vars: dict[int, gp.Var] = {}  # unit penalty vars
        self.mach_assign: dict[tuple[int, int], gp.Var] = {}  # machine assignment vars
        self.op_assign: dict[tuple[int, int], gp.Var] = {}  # operator assignment vars

    def generate_problem(self) -> None:
        """Generates the Gurobi problem."""

        assert self.inst.I is not None
        assert self.inst.M is not None
        assert self.inst.O is not None
        assert self.inst.alpha is not None
        assert self.inst.beta is not None

        print(f"Generating Gurobi problem for {self.inst.name}...")

        self.m = gp.Model(self.inst.name)

        print("Greedy solving for time horizon estimation")
        self.inst.greedy_solve()
        T = int(max([j.C() for j in self.inst.jobs.values()]) * 1.25)

        B_vars: dict[int, gp.Var] = {}
        C_vars: dict[int, gp.Var] = {}
        T_vars: dict[int, gp.Var] = {}
        U_vars: dict[int, gp.Var] = {}

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
                self.m.addConstr(
                    B_vars[job.S[idx]] >= C_vars[job.S[idx - 1]]
                )  # B_i >= C_{i-1}

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

        running_after_B = self.m.addVars(
            range(1, T + 1), range(1, self.inst.I + 1), vtype=GRB.BINARY
        )
        running_before_C = self.m.addVars(
            range(1, T + 1), range(1, self.inst.I + 1), vtype=GRB.BINARY
        )
        running = self.m.addVars(
            range(1, T + 1), range(1, self.inst.I + 1), vtype=GRB.BINARY
        )

        for t, tid in running_after_B:
            self.m.addConstr((running_after_B[t, tid] == 0) >> (t <= B_vars[tid] - 1))
            self.m.addConstr((running_after_B[t, tid] == 1) >> (t >= B_vars[tid]))
        for t, tid in running_before_C:
            self.m.addConstr((running_before_C[t, tid] == 0) >> (t >= C_vars[tid]))
            self.m.addConstr((running_before_C[t, tid] == 1) >> (t <= C_vars[tid] - 1))
        for t, tid in running:
            self.m.addConstr(
                running[t, tid]
                == gp.and_(running_after_B[t, tid], running_before_C[t, tid])
            )

        print("Creating machines and operators task assignments tables...")

        mach_assign: dict[tuple[int, int], gp.Var] = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                mids = {worker.mid for worker in task.workers}
                for mid in mids:
                    mach_assign[tid, mid] = self.m.addVar(
                        name=f"task_{tid}_machine_{mid}", vtype=GRB.BINARY
                    )
                self.m.addConstr(
                    cast(
                        gp.TempLConstr,
                        sum([mach_assign[tid, mid] for mid in mids]) == 1,
                    )
                )

        oper_assign: dict[tuple[int, int], gp.Var] = {}
        for job in self.inst.jobs.values():  # iterate over jobs
            for tid in job.S:  # iterate over tasks
                task = self.inst.tasks[tid]
                oids = {worker.oid for worker in task.workers}
                for oid in oids:
                    oper_assign[tid, oid] = self.m.addVar(
                        name=f"task_{tid}_operator_{oid}", vtype=GRB.BINARY
                    )
                self.m.addConstr(
                    cast(
                        gp.TempLConstr,
                        sum([oper_assign[tid, oid] for oid in oids]) == 1,
                    )
                )

        print("Creating machines and operators business tables...")

        mach_business = self.m.addVars(
            range(1, T + 1), range(1, self.inst.M + 1), vtype=GRB.INTEGER
        )
        oper_business = self.m.addVars(
            range(1, T + 1), range(1, self.inst.O + 1), vtype=GRB.INTEGER
        )
        for t, mid in mach_business:
            self.m.addConstr(mach_business[t, mid] <= 1)
        for t, oid in oper_business:
            self.m.addConstr(oper_business[t, oid] <= 1)

        for t, mid in mach_business:
            assigned_and_running = []
            for tid in [k[0] for k in mach_assign if k[1] == mid]:
                assigned_and_running.append(self.m.addVar(vtype=GRB.INTEGER))
                self.m.addConstr(
                    assigned_and_running[-1]
                    == gp.and_(mach_assign[tid, mid], running[t, tid])
                )
            self.m.addConstr(mach_business[t, mid] == sum(assigned_and_running))

        for t, oid in oper_business:
            assigned_and_running = []
            for tid in [k[0] for k in oper_assign if k[1] == oid]:
                assigned_and_running.append(self.m.addVar(vtype=GRB.INTEGER))
                self.m.addConstr(
                    assigned_and_running[-1]
                    == gp.and_(oper_assign[tid, oid], running[t, tid])
                )
            self.m.addConstr(oper_business[t, oid] == sum(assigned_and_running))

        print("Adding objective function...")

        w = [job.w for job in self.inst.jobs.values()]  # job weights
        JC_vars = [
            C_vars[job.S[-1]] for job in self.inst.jobs.values()
        ]  # job completion dates
        self.m.setObjective(
            sum(
                [
                    wj * (Cj + self.inst.alpha * Uj + self.inst.beta * Tj)
                    for wj, Cj, Uj, Tj in zip(
                        w, JC_vars, U_vars.values(), T_vars.values()
                    )
                ]
            ),
            GRB.MINIMIZE,
        )

        # store variables
        self.B_vars, self.C_vars, self.T_vars, self.U_vars = (
            B_vars,
            C_vars,
            T_vars,
            U_vars,
        )
        self.mach_assign, self.oper_assign = mach_assign, oper_assign

        print("Gurobi problem generated.")

    def warmup(self) -> None:
        """Sets intial values for variables to solve with warm start."""

        assert self.m is not None

        # set greedy beginning times
        for k, v in self.B_vars.items():
            beginning = self.inst.tasks[k].B
            assert beginning is not None
            v.Start = beginning
        # set greedy completion times
        for k, v in self.C_vars.items():
            completion = self.inst.tasks[k].C
            assert completion is not None
            v.Start = completion
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

    def solve(self) -> None:
        """Solves the PuLP problem."""

        assert self.m is not None
        self.m.optimize()

    def show_status(self) -> None:
        assert self.m is not None
        sc = gp.StatusConstClass
        status_codes = {
            sc.__dict__[k]: k for k, v in sc.__dict__.items() if isinstance(v, int)
        }
        print(
            f"Problem status: {status_codes[self.m.status]}\nObective value: {self.m.objVal}"
        )

    def savefile(self) -> None:
        """Saves the problem to disk as Mathematical Programming System file."""

        assert self.m is not None
        path = f"lp_problems/gurobi_{self.inst.name}.mps"
        try:
            self.m.write(path)
            print(f"Problem saved to {path}")
        except (OSError, ValueError, RuntimeError):
            print("Failed saving to file!")


def export_to_dot_separated(inst: Instance, tasks: list[int], filename: str) -> None:
    with open(filename, "w") as f:
        f.write("graph InstanceGraph {\n")  # Using directed graph
        f.write('bgcolor="lightgray"\n')
        f.write("overlap=false;\n")  # Prevents node overlap
        f.write("splines=true;\n")  # Makes edges curved for better readability
        f.write("nodesep=1.0;\n")  # Increases spacing between nodes
        f.write("ranksep=0.8;\n")  # Increases vertical separation

        # Create nodes
        for t in tasks:
            color = "lightblue"
            j_of_t = inst.tasks[t].j
            f.write(
                f'    {t} [label="T{t}\n(J{j_of_t})", shape="ellipse", style="filled", fillcolor="{color}"];\n'
            )

        for t1_id, t2_id in combinations(tasks, 2):
            if inst.tasks[t1_id].j == inst.tasks[t2_id].j:
                continue
            # Create edges for shared machines (undirected)
            # compute the intersection of their machines
            w1 = {w.mid for w in inst.tasks[t1_id].workers}
            w2 = {w.mid for w in inst.tasks[t2_id].workers}
            shared_machines = w1.intersection(w2)

            f.writelines(
                f'    {t1_id} -- {t2_id} [label=M{m},color="blue"];\n'
                for m in shared_machines
            )

            # Create edges for shared individual operators (undirected)
            # compute the intersection of their operators
            w1 = {w.oid for w in inst.tasks[t1_id].workers}
            w2 = {w.oid for w in inst.tasks[t2_id].workers}
            shared_operators = w1.intersection(w2)
            f.writelines(
                f'    {t1_id} -- {t2_id} [label=O{o}, color="red"];\n'
                for o in shared_operators
            )
        f.write("}\n")


def export_to_dot_pairs(inst: Instance, tasks: list[int], filename: str) -> None:
    with open(filename, "w") as f:
        f.write("graph InstanceGraph {\n")  # Using directed graph
        f.write('bgcolor="lightgray"\n')
        f.write("overlap=false;\n")  # Prevents node overlap
        f.write("splines=true;\n")  # Makes edges curved for better readability
        f.write("nodesep=1.0;\n")  # Increases spacing between nodes
        f.write("ranksep=0.8;\n")  # Increases vertical separation

        # Create nodes
        for t in tasks:
            color = "lightgreen"
            j_of_t = inst.tasks[t].j
            f.write(
                f'    {t} [label="T{t}\n(J{j_of_t})", shape="ellipse", style="filled", fillcolor="{color}"];\n'
            )

        # Create edges for shared individual workers (undirected)
        for t1_id, t2_id in combinations(tasks, 2):
            if inst.tasks[t1_id].j == inst.tasks[t2_id].j:
                continue
            # compute the intersection of their workers
            w1 = {(w.mid, w.oid) for w in inst.tasks[t1_id].workers}
            w2 = {(w.mid, w.oid) for w in inst.tasks[t2_id].workers}
            shared_workers = w1.intersection(w2)

            f.writelines(
                f'    {t1_id} -- {t2_id} [label=M{sw[0]}O{sw[1]}, color="black"];\n'
                for sw in shared_workers
            )
        f.write("}\n")


def export_to_dot_sets(inst: Instance, tasks: list[int], filename: str) -> None:
    with open(filename, "w") as f:
        f.write("graph InstanceGraph {\n")  # Using directed graph
        f.write('bgcolor="lightgray"\n')
        f.write("overlap=false;\n")  # Prevents node overlap
        f.write("splines=true;\n")  # Makes edges curved for better readability
        f.write("nodesep=1.0;\n")  # Increases spacing between nodes
        f.write("ranksep=0.8;\n")  # Increases vertical separation

        # Create nodes
        for t in tasks:
            color = "lightgreen"
            j_of_t = inst.tasks[t].j
            f.write(
                f'    {t} [label="T{t}\n(J{j_of_t})", shape="ellipse", style="filled", fillcolor="{color}"];\n'
            )

        edges_ma_dicts = {}
        edges_op_dicts = {}
        for t1_id, t2_id in combinations(tasks, 2):
            if inst.tasks[t1_id].j == inst.tasks[t2_id].j:
                continue
            # Create edges for shared machines (undirected)
            # compute the intersection of their machines
            w1 = {w.mid for w in inst.tasks[t1_id].workers}
            w2 = {w.mid for w in inst.tasks[t2_id].workers}
            shared_machines = w1.intersection(w2)
            edges_ma_dicts[(t1_id, t2_id)] = shared_machines

            # Create edges for shared individual operators (undirected)
            # compute the intersection of their operators
            w1 = {w.oid for w in inst.tasks[t1_id].workers}
            w2 = {w.oid for w in inst.tasks[t2_id].workers}
            shared_operators = w1.intersection(w2)
            edges_op_dicts[(t1_id, t2_id)] = shared_operators

        for (t1_id, t2_id), sm in edges_ma_dicts.items():
            if len(sm) > 0:
                f.write(
                    f'    {t1_id} -- {t2_id} [label="M{sm!s}", color="blue", penwidth={len(sm)}];\n'
                )
        for (t1_id, t2_id), so in edges_op_dicts.items():
            if len(so) > 0:
                f.write(
                    f'    {t1_id} -- {t2_id} [label="O{so!s}", color="red", penwidth={len(so)}];\n'
                )
        f.write("}\n")


def export_incompatibilities(
    inst: Instance, tasks: list[int], filename: str, display_tasks: bool = False
) -> None:
    with open(filename, "w") as f:
        f.write("graph InstanceGraph {\n")  # Using directed graph
        f.write('bgcolor="lightgray"\n')
        f.write("overlap=false;\n")  # Prevents node overlap
        f.write("splines=true;\n")  # Makes edges curved for better readability
        f.write("nodesep=1.0;\n")  # Increases spacing between nodes
        f.write("ranksep=0.8;\n")  # Increases vertical separation

        # Create nodes of workers
        worker_nodes = set()
        for t in tasks:
            for w in inst.tasks[t].workers:
                combination = (w.mid, w.oid)
                if combination not in worker_nodes:
                    h = hash(combination)
                    worker_nodes.add(combination)
                    f.write(
                        f'    {h} [label="M{w.mid}O{w.oid}", shape="ellipse", style="filled", fillcolor="lightblue"];\n'
                    )

        # Create edges for incompatible workers
        edges = set()
        for w1, w2 in combinations(worker_nodes, 2):
            if w1[0] == w2[0] or w1[1] == w2[1]:
                # Create edges for these workers
                edges.add((w1, w2))
                h1 = hash(w1)
                h2 = hash(w2)
                f.write(f'    {h1} -- {h2} [color="blue"];\n')
        f.write("}\n")

        # create nodes of tasks
        if display_tasks:
            for t in tasks:
                j_of_t = inst.tasks[t].j
                f.write(
                    f'    {hash(t)} [label="T{t}\n(J{j_of_t})", shape="ellipse", style="filled", fillcolor="orange"];\n'
                )

        # Create edges for workers-tasks assignments
        if display_tasks:
            for t in tasks:
                j_of_t = inst.tasks[t].j
                for w in inst.tasks[t].workers:
                    combination = (w.mid, w.oid)
                    f.write(
                        f'    {hash(t)} -- {hash(combination)} [color="orangered"];\n'
                    )
