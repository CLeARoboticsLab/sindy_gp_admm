using GaussianSINDy
using GaussianProcesses
using Plots 
using Optim 

# choose ODE, plot states --> measurements 
fn = predator_prey 
x0, dt, t, x, dx_true, dx_fd = ode_states(fn, 0, 2) 

dx_noise = 0.1 

dx_fd = dx_true + dx_noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
dx_fd = dx_fd[:,1] ; dx_true = dx_true[:,1] 

# dx_fd = sin.(t) + 0.05*randn(length(t));   #regressors

## ============================================ ## 
# GP! NO hp optimization 

σ_f = log(1.0) ; l = log(1.0) ; σ_n = log(0.1) 

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = SE( σ_f, l ) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = σ_n ;              # (optional) log std dev of obs noise 

# fit GP 
# y_train = dx_train - Θx*ξ   
x_train = t 
y_train = dx_fd
gp      = GP(x_train, y_train, mZero, kern, log_noise) 

# tests 
x_test  = t 
μ, σ²   = predict_y( gp, x_test )
μ_post, Σ_post = post_dist( x_train, y_train, x_test, exp(σ_f), exp(l), exp(σ_n) ) 
σ²_post = diag( Σ_post ) 

a = Animation()

plt = plot(gp; xlabel="x", ylabel="y", title="Gaussian Process (no HP opt)", label = "gp toolbox", legend = :outerright, size = [800 300] ) 
    frame(a, plt) 
plot!( plt, t, dx_true, label = "true", c = :green ) 
    frame(a, plt) 
plot!( plt, x_test, μ, label = "gp predict", c = :red, ls = :dash, ribbon = ( μ - σ², μ + σ² ) )
    frame(a, plt) 
plot!( plt, x_test, μ_post, label = "manual", ls = :dash, c = :cyan, lw = 1.5, ribbon = ( μ_post - σ²_post, μ_post + σ²_post ) )
    frame(a, plt) 

g = gif(a, fps = 0.75) 
display(g) 
display(plt) 

## ============================================ ##
# hp optimization (toolbox) 

a = Animation()

# toolbox 
@time result = optimize!(gp) 
plt = plot( gp, title = "Gaussian Process (Opt HPs)", label = "gp toolbox", legend = :outerright, size = [800 300] ) 
    frame(a, plt) 
plot!( plt, t, dx_true, label = "true", c = :green )
    frame(a, plt) 
μ_opt, σ²_opt = predict_y( gp, x_test )
plot!( plt, x_test, μ_opt, label = "gp predict", c = :red, ls = :dash, ribbon = ( μ_opt - σ²_opt, μ_opt + σ²_opt ) ) 
    frame(a, plt) 

# ----------------------- # 
# hp optimization (June) --> post mean  

@time μ_post, Σ_post, hp_post = post_dist_hp_opt( x_train, y_train, x_test )
σ²_manual = diag( Σ_post ) 
plot!( plt, x_test, μ_post, label = "manual", c = :cyan, ls = :dashdot, ribbon = ( μ_post - σ²_manual, μ_post + σ²_manual )  )
    frame(a, plt) 

g = gif(a, fps = 0.75)
display(g) 
display(plt) 

println( "gp toolbox opt hp = ", exp.( result.minimizer ) ) 
println( "manual opt hp     = ", hp_post ) 

## ============================================ ##
# try using optimized hps 

result  = optimize!(gp) 

σ_f = result.minimizer[1] 
l   = result.minimizer[2] 
σ_n = result.minimizer[3] 
hp  = [σ_f, l, σ_n] 

# kernel  
mZero     = MeanZero() ;            # zero mean function 
kern      = SE( σ_f, l ) ;        # squared eponential kernel (hyperparams on log scale) 
log_noise = σ_n ;              # (optional) log std dev of obs noise 

# fit GP 
# y_train = dx_train - Θx*ξ   
x_train = t 
y_train = dx_fd
gp      = GP(x_train, y_train, mZero, kern, log_noise) 

# tests 
x_test  = t 
μ, σ²   = predict_y( gp, x_test )

plt = plot(gp; xlabel="x", ylabel="y", title="Gaussian Process (HP opt)", label = "gp toolbox", legend = :outerright, size = [800 300] ) 
    frame(a, plt) 
plot!( plt, t, dx_true, label = "true", c = :green ) 
    frame(a, plt) 
plot!( plt, x_test, μ, label = "gp predict", c = :red, ls = :dash, ribbon = ( μ - σ², μ + σ² ) )
    frame(a, plt) 

g = gif(a, fps = 0.75)
display(g) 
display(plt) 



