
struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

using GaussianSINDy
using LinearAlgebra 
using Plots 
using Dates 
using Optim 
using GaussianProcesses

## ============================================ ##
# choose ODE, plot states --> measurements 

#  
fn             = predator_prey 
plot_option    = 0 
savefig_option = 0 
fd_method      = 2 # 1 = forward, 2 = central, 3 = backward 

# choose ODE, plot states --> measurements 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, plot_option, fd_method) 

# truth coeffs 
Ξ_true = SINDy_test( x, dx_true, 0.1 ) 
Ξ_true = Ξ_true[:,1] 
 
dx_noise  = 1.0 

# ----------------------- #
# MONTE CARLO GPSINDY 

    dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 

## ============================================ ##
## ============================================ ##
## ============================================ ##
# sindy_gp_admm 

    # ----------------------- #
    # SINDy 
    
    λ          = 0.2 
    n_vars     = size(x, 2) 
    poly_order = n_vars 

    Ξ_true  = SINDy_test( x, dx_true, λ ) 
    Ξ_sindy = SINDy_test( x, dx_fd, λ ) 

    # SINDy  
    Θx = pool_data_test(x, n_vars, poly_order) 
    Ξ  = sparsify_dynamics_test(Θx, dx_fd, λ, n_vars) 

    n = size(Ξ, 1)

    # ----------------------- #
    # loop with state j

    hist_nvars = [] 

    j = 1 
    println( "j = ", j )
    # for j = 1 : n_vars
    
        hist = Hist( [], [], [], [], [], [], [], [] )  

        # initial loss function vars 
        dx = dx_fd[:,j] 

        # assign for f_hp_opt 
        f_hp(ξ, (σ_f, l, σ_n)) = f_obj( σ_f, l, σ_n, dx, ξ, Θx )

        # l1 norm 
        g(z) = λ * sum(abs.(z)) 

        # augmented Lagrangian (scaled form) 
        ρ = 1.0 
        aug_L(ξ, hp, z, u) = f_hp(ξ, hp) + g(z) + ρ/2 .* norm( ξ - z + u )^2         

        # ----------------------- # 
        # LASSO ADMM GP OPT 

        # define constants 
        max_iter = 1000  
        abstol   = 1e-2 
        reltol   = 1e-2           # save matrix-vector multiply 

        # ADMM solver 
        ξ = z = u = zeros(n) 

        # counter 
        iter = 0 
        
        # ----------------------- # 
        # ξ-update (optimization) 

        hp = log.( [ 1.0, 1.0, 0.1 ] )
        ξ = opt_ξ( aug_L, 0*ξ, z, u, hp ) 
        println( "ξ = ", ξ ) 

## ============================================ ##

    # for k = 1:max_iter 

        # increment counter 
        iter += 1 
        println( "iter = ", iter )

        z_old = z 

        # ADMM LASSO! 
        ξ, z, u, hp = admm_lasso(t, dx, Θx, ξ, z, u, aug_L, λ, true )     
        σ_f = hp[1] ; l = hp[2] ; σ_n = hp[3]    

        println( "ξ = ", ξ )
        println( "z = ", z )
        println( "hp = ", hp )

        p = f_hp(ξ, hp) + g(z)   
        push!( hist.objval, p )
        push!( hist.fval, f_hp( ξ, hp ) )
        push!( hist.gval, g(z) ) 
        push!( hist.hp, hp )
        push!( hist.r_norm, norm(ξ - z) )
        push!( hist.s_norm, norm( -ρ*(z - z_old) ) )
        push!( hist.eps_pri, sqrt(n)*abstol + reltol*max(norm(ξ), norm(-z)) ) 
        push!( hist.eps_dual, sqrt(n)*abstol + reltol*norm(ρ*u) ) 

        if hist.r_norm[end] < hist.eps_pri[end] && hist.s_norm[end] < hist.eps_dual[end] 
            println("converged!")  
            println( "gpsindy err = ", norm( Ξ_true[:,j] - z ) ) 
            println( "sindy err   = ", norm( Ξ_true[:,j] - Ξ_sindy[:,j] ) ) 
            push!(hist_nvars, hist)
        end 

    # end 
    
## ============================================ ##
# back to MONTE CARLO GP SINDy

    Ξ_sindy_err   = [ norm( Ξ_true[:,1] - Ξ_sindy[:,1] ), norm( Ξ_true[:,2] - Ξ_sindy[:,2] )  ] 
    z_gpsindy_err = [ norm( Ξ_true[:,1] - z_gpsindy[:,1] ), norm( Ξ_true[:,2] - z_gpsindy[:,2] )  ] 

    
