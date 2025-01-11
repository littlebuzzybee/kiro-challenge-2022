using JuMP
# using Ipopt
using Gurobi
# using SCIP
# using KNITRO

# Import the model

model = read_from_file("model.mps")
print(model) # beware if the model is too large

# Select the solver and optimize the model

set_optimizer(model, Gurobi.Optimizer)
optimize!(model)
