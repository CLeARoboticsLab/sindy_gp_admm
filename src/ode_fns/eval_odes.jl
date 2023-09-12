using DifferentialEquations


## ============================================ ##
# build dx_fn from Ξ and integrate 

export dx_Ξ_integrate 
function dx_Ξ_integrate( data_train, data_test, Ξ, x0_train_GP, x0_test_GP ) 
    
    # get sizes 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

    # build dx_fn from Ξ and integrate 
    dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ ) 
    x_train_sindy = integrate_euler( dx_fn_sindy, x0_train_GP, data_train.t, data_train.u ) 
    x_test_sindy  = integrate_euler( dx_fn_sindy, x0_test_GP, data_test.t, data_test.u ) 

    return x_train_sindy, x_test_sindy 
end 


## ============================================ ##
# standardize data_train and data_test 

export ode_stand_data 
function ode_stand_data( data_train, fn, p, noise ) 

    # ----------------------- # 
    # standardize  
    x_noise  = stand_data(data_train.t, data_train.x_noise)
    x_true   = stand_data(data_train.t, data_train.x_true)
    dx_true  = dx_true_fn(data_train.t, x_true, p, fn)
    dx_noise = data_train.dx_true + noise * randn( size(dx_true, 1), size(dx_true, 2) )

    data_train = data_struct( data_train.t, data_train.u, x_true, dx_true, x_noise, dx_noise ) 

    return data_train 
end 


## ============================================ ##

export ode_train_test 
function ode_train_test( fn, noise = 0.01 ) 

    x0, dt, t, x_true, dx_true, dx_fd, p, u = ode_states(fn, 0, 2) 

    # noise 
    x_noise  = x_true + noise*randn( size(x_true, 1), size(x_true, 2) )
    dx_noise = dx_true + noise*randn( size(dx_true, 1), size(dx_true, 2) ) 
    
    # split into training and test data 
    test_fraction = 0.2 
    portion       = 5 
    if u == false 
        u_train = false ; u_test  = false 
    else 
        u_train, u_test = split_train_test(u, test_fraction, portion) 
    end 
    t_train,        t_test        = split_train_test(t,        test_fraction, portion) 
    x_train_true,   x_test_true   = split_train_test(x_true,   test_fraction, portion) 
    dx_train_true,  dx_test_true  = split_train_test(dx_true,  test_fraction, portion) 
    x_train_noise,  x_test_noise  = split_train_test(x_noise,  test_fraction, portion) 
    dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion) 

    data_train = data_struct( t_train, u_train, x_train_true, dx_train_true, x_train_noise, dx_train_noise ) 
    data_test  = data_struct( t_test, u_test, x_test_true, dx_test_true, x_test_noise, dx_test_noise) 

    return data_train, data_test 
end 


## ============================================ ##

export solve_ode 
function solve_ode(fn, x0, str, p, ts, dt, plot_option)

    # solve ODE 
    prob = ODEProblem(fn, x0, ts, p) 
    sol  = solve(prob, saveat = dt) 

    # extract variables --> measurements 
    sol_total = sol 
    x = sol.u ; x = mapreduce(permutedims, vcat, x) 
    t = sol.t 

    # get control inputs (if they exist) 
    u = fn_control_inputs( fn, t ) 

    if plot_option == 1 
        plot_dyn(t, x, str)
    end 

    return t, x, u 

end 


## ============================================ ##

export ode_states 
function ode_states(fn, plot_option, fd_method)

    x0, str, p, ts, dt = init_params(fn) 
    t, x, u = solve_ode(fn, x0, str, p, ts, dt, plot_option) 

    # ----------------------- #
    # derivatives 
    dx_fd   = fdiff(t, x, fd_method)    # finite difference 
    dx_true = dx_true_fn(t, x, p, fn)   # true derivatives 

    # plot derivatives 
    if plot_option == 1 
        plot_deriv(t, dx_true, dx_fd, dx_tv, str) 
    end 

    return x0, dt, t, x, dx_true, dx_fd, p, u 

end 


## ============================================ ##

export validate_data 
function validate_data(t_test, xu_test, dx_fn, dt)


    n_vars = size(xu_test, 2) 
    x0     = [ xu_test[1] ] 
    if n_vars > 1 
        x0 = xu_test[1,:] 
    end 

    # dt    = t_test[2] - t_test[1] 
    tspan = (t_test[1], t_test[end])
    prob  = ODEProblem(dx_fn, x0, tspan) 

    # solve the ODE
    sol   = solve(prob, saveat = dt)
    # sol = solve(prob,  reltol = 1e-8, abstol = 1e-8)
    x_validate = sol.u ; 
    x_validate = mapreduce(permutedims, vcat, x_validate) 
    t_validate = sol.t 

    return t_validate, x_validate 

end 

## ============================================ ##
# function to check Inf --> NaN 

export inf_nan 
function inf_nan( x )
    
    for i in 1:length(x) 
        if x[i] == Inf || x[i] == -Inf 
            x[i] = NaN   
        end 
    end    

    return x 
end 

## ============================================ ##
# (5) Function to integrate an ODE using forward Euler integration.

export integrate_euler 
function integrate_euler(dx_fn, x0, t, u = false)
    # TODO: Euler integration consists of setting x(t + δt) ≈ x(t) + δt * ẋ(t, x(t), u(t)).
    #       Returns x(T) given x(0) = x₀.

    xt = x0 
    z  = zeros(size(x0, 1)) 
    # dt = t[2] - t[1] 
    n  = length(t) 

    x_hist = [] 
    # t_hist = [] 
    push!( x_hist, xt ) 
    # push!( t_hist, t[1] ) 
    if u == false 

        for i = 1 : n - 1 
            dt  = t[i+1] - t[i] 
            xt  = inf_nan( xt )
            xt += dt * dx_fn( xt, 0, t[i] ) 
            push!( x_hist, xt ) 
            # push!( t_hist, t[1] + i * dt ) 
        end     

    else 

        u_vars = size(u, 2) 

        for i = 1 : n - 1 

            dt  = t[i+1] - t[i] 
            xt  = inf_nan( xt )
            xut = copy( xt ) 
            if u_vars > 1 
                for j = 1 : u_vars 
                    push!( xut, u[i,j] ) 
                end 
            else
                ut = u[i]  
                push!( xut, ut ) 
            end 
            xt += dt * dx_fn( xut, 0, t[i] ) 
            push!( x_hist, xt ) 
            # push!( t_hist, t[1] + i * dt ) 

        end 
    
    end 

    x_hist = vv2m(x_hist) 

    return x_hist 
end 

## ============================================ ##
# (5) Function to integrate an ODE using forward Euler integration.

export integrate_euler_unicycle 
function integrate_euler_unicycle(fn, x0, t, u = false)
    # TODO: Euler integration consists of setting x(t + δt) ≈ x(t) + δt * ẋ(t, x(t), u(t)).
    #       Returns x(T) given x(0) = x₀.

    xt = x0 
    ut = u[1] 
    z  = zeros(size(x0, 1)) 
    # dt = t[2] - t[1] 
    n  = length(t) 

    x_hist = [] 
    push!( x_hist, xt ) 
    if u == false 

        for i = 1 : n - 1 
            dt  = t[i+1] - t[i] 
            xt += dt * fn( z, xt, 0, t[i] ) 
            push!( x_hist, xt ) 
            # push!( t_hist, t[1] + i * dt ) 
        end     

    else 

        u_vars = size(u, 2) 
        for i = 1 : n - 1 
            dt  = t[i+1] - t[i] 
            ut = u[i,:] 
            xt += dt * fn( z, xt, 0, t[i], ut ) 
            push!( x_hist, xt ) 
            # push!( t_hist, t[1] + i * dt ) 
        end 
    
    end 

    x_hist = vv2m(x_hist) 

    return x_hist 
end

## ============================================ ##

export dx_true_fn 
function dx_true_fn(t, x, p, fn)

    # true derivatives 
    dx_true = 0*x
    n_vars  = size(x, 2) 
    z       = zeros(n_vars) 

    for i = 1 : length(t) 
        dx_true[i,:] = fn( z, x[i,:], p, t[i] ) 
    end 

    return dx_true 

end 

## ============================================ ##

export build_dx_fn 
function build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

    n_vars = x_vars + u_vars 

    # define pool_data functions 
    fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

    # numerically evaluate each function at x and return a vector of numbers
    𝚽( xu, fn_vector ) = [ f(xu) for f in fn_vector ]

    # create vector of functions, each element --> each state 
    dx_fn_vec = Vector{Function}(undef,0) 
    for i = 1 : x_vars 
        # define the differential equation 
        push!( dx_fn_vec, (xu,p,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
    end 

    dx_fn(xu,p,t) = [ f(xu,p,t) for f in dx_fn_vec ] 

    return dx_fn 

end 

## ============================================ ##




