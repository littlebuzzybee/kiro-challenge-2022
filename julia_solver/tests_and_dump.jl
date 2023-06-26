### BROUILLON


# survey les collisions de couples opérateurs_machines et faire un histogramme pour ensuite optimiser le code
intersection = Float64[];
for i in 1:nb_tasks
    for j in i+1:nb_tasks
        push!(intersection, norm(compat_machine_operator_per_task[i,:,:] .&&  compat_machine_operator_per_task[j,:,:], 1));
    end
end


using Plots;
histogram(intersection)
# on constate que dans une immense majorité de cas, les tâches n'ont quasi aucun couple possible opérateur-machine en commun.
# Ceci veut dire que "souvent" on n'aura pas à se préoccuper de telles intersections pour assigner les tâches avec les ressources disponibles et donc qu'on peut simplement essayer de parcourir le produit cartésien des couples possibles à chaque tâche à assigner lors de l'étape de temps en supprimant les solutions qui collisionnent.




using CUDA;

function inplace_kernel!(vec::CuDeviceVector{ComplexF32}, α::ComplexF32, ϵ::Float32, maxiter::Int32, res::CuDeviceVector{Float16})
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(vec) # verify if the index i actually corresponds to a index of the vectors
        ind = Int32(0);
        iterexp = 3.f0*vec[i];
        while (CUDA.abs(iterexp) > ϵ && ind < maxiter) # dummy algorithm to illustrate kernel writing
            iterexp *= α;
            vec[i] += CUDA.exp(-CUDA.abs(2.0f0*vec[i]));
            ind += 1;
        end
        res[i] = convert(Float16, CUDA.abs(iterexp));
    end
    return nothing;
end
