struct Hist 
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using DifferentialEquations 
using GaussianSINDy
using LinearAlgebra 
using ForwardDiff 
using Optim 
using Plots 
using CSV 
using DataFrames 
using Symbolics 
using PrettyTables 
using Test 
using NoiseRobustDifferentiation
using Random, Distributions 


## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn          = predator_prey
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 


## ============================================ ##
# SINDy alone 

λ = 0.1 
n_vars     = size(x, 2) 
poly_order = n_vars 

Ξ_true = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 
Ξ_fd   = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 


## ============================================ ##
# split into training and validation data 

train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 


## ============================================ ##
# SINDy + GP + ADMM 

# # truth 
# hist_true = Hist( [], [], [], [], [] ) 
# @time z_true, hist_true = sindy_gp_admm( x, dx_true, λ, hist_true ) 
# display(z_true) 

λ = 0.02 

# finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
@time z_fd, hist_fd = sindy_gp_admm( x_train, dx_fd_train, λ, hist_fd ) 
display(z_fd) 

## ============================================ ##
# validation 

using Symbolics 

@variables x1 x2 
x_sym = [x1 x2]
Θ_sym = pool_data(x_sym, 2, 2) 

dx1 = sum(Θ_sym * z_fd[:,1]) 
dx1 = build_function(dx1, x1, x2, expression = Val{false})

dx2 = sum(Θ_sym * z_fd[:,2])
dx2 = build_function(dx2, x1, x2, expression = Val{false})

dx = [0; 0] 
dx[1] = dx1 
dx[2] = dx2 

function gpsindy( dx1_fn, dx2_fn )
    dx[1] = dx1 
    dx[2] = dx2 
    return dx 
end 

## ============================================ ##
# validation 

# display training data 

n_vars = size(x,2) 

# construct empty vector for plots 
plot_vec_x = [] 
for i = 1:n_vars 
    plt = plot(t_train, x_train[:,i], title = "State $(i)", legend = false, lw = 2)
    plot!(t_test, x_test[:,i], c = :red )
    push!(plot_vec_x, plt)
end 

plot_x = plot(plot_vec_x ..., 
    layout = (n_vars,1), 
    size = [600 n_vars*300], 
    xlabel = "Time (s)", 
    plot_title = "Training vs. Validation" ) 
display(plot_x) 

