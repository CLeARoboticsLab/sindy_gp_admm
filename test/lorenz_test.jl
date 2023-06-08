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
using Infiltrator


## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn          = predator_prey 
plot_option = 1 
t, x, dx_true, dx_fd = ode_states(fn, plot_option) 

# split into training and validation data 
train_fraction = 0.7 
t_train, t_test             = split_train_test(t, train_fraction) 
x_train, x_test             = split_train_test(x, train_fraction) 
dx_true_train, dx_true_test = split_train_test(dx_true, train_fraction) 
dx_fd_train, dx_fd_test     = split_train_test(dx_fd, train_fraction) 


## ============================================ ##

# λ = 0.1 
# n_vars     = size(x, 2) 
# poly_order = n_vars 

# # SINDy alone 
# Ξ_true = SINDy_c_recursion(x, dx_true, 0, λ, poly_order ) 
# Ξ_fd   = SINDy_c_recursion(x, dx_fd, 0, λ, poly_order ) 

# ## ============================================ ##

λ = 0.1 

u_train = 2sin.(t_train) + 2sin.(t_train/10) ; 
Ξ_true  = SINDy_c( x_train, u_train, dx_true_train, λ )
Ξ_fd    = SINDy_c( x_train, u_train, dx_fd_train, λ )

λ = 0.01 

## ============================================ ##

# SINDy + GP + ADMM - finite difference 
hist_fd = Hist( [], [], [], [], [] ) 
# @time z_fd, hist_fd = sindy_gp_admm( x_train, dx_true_train, λ, hist_fd ) 
@time z_fd, hist_fd = sindyc_gp_admm( x_train, u_train, dx_fd_train, λ, hist_fd ) 
display(z_fd) 


## ============================================ ##
# generate + validate data 

# validate_plot_data( t_train, x_train, t_test, x_test, z_fd, Ξ_fd, poly_order, 1 ) 

n_vars = size( x,2 ) + size(u_train,2 )
poly_order = size(x,2) 
dx_gpsindy_fn = build_dx_fn(x_vars, n_vars, poly_order, z_fd) 
dx_sindy_fn   = build_dx_fn(x_vars, n_vars, poly_order, Ξ_fd)

## ============================================ ##

# with forcing 
u_test = 2sin.(t_test) + 2sin.(t_test/10) ; 

t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, u_test, dx_gpsindy_fn)
t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, u_test, dx_sindy_fn)

## ============================================ ##

plot_font = "Computer Modern" 
default(
    fontfamily = plot_font,
    linewidth = 2, 
    # framestyle = :box, 
    label = nothing, 
    grid = false, 
    )
# scalefontsizes(1/1.3)

ptitles = ["Prey", "Predator"]

plot_vec = [] 
for i = 1:n_vars 

    # display training data 
    p = plot(t_train, x_train[:,i], 
        lw = 3, 
        c = :gray, 
        label = "train (70%)", 
        grid = false, 
        xlim = (t_train[end]*3/4, t_test[end]), 
        # legend = :outerbottom , 
        legend = false , 
        xlabel = "Time (s)", 
        title  = string(ptitles[i],", x$(i)"), 
        xticks = 0:2:10, 
        yticks = 0:0.5:4,     
        ) 
    plot!(t_test, x_test[:,i], 
        ls = :dash, 
        c = :blue,
        lw = 3,  
        label = "test (30%)" 
        )
    plot!(t_gpsindy_val, x_gpsindy_val[:,i], 
        ls = :dash, 
        lw = 1.5, 
        c = :red, 
        label = "GP SINDy" 
        )
    # plot!(t_sindy_val, x_sindy_val[:,i], 
    #     ls = :dashdot, 
    #     lw = 1.5, 
    #     c = :green, 
    #     label = "SINDy" 
    #     )

    push!( plot_vec, p ) 

end 
# plot!(legend = false)

p_train_val = plot(plot_vec ... , 
    layout = (1, n_vars), 
    size = [ n_vars*600 600 ], 
    plot_title = "Training vs. Validation Data", 
    # titlefont = font(16), 
    )
display(p_train_val) 



# plot_dx = plot(plot_vec_dx ... , 
# layout = (n_vars, 1), 
# size = [600 n_vars*300], 
# plot_title = "Derivatives. ODE fn = $( str )" )
# display(plot_dx) 

## ============================================ ##

savefig("./plot.png")

