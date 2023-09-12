using GaussianSINDy 

# generate data 
fn = predator_prey 

if fn == predator_prey 
    stand_data_option = 1 
elseif fn == unicycle 
    stand_data_option = 0  
end 

# set up noise vec 
noise_vec      = []
noise_vec_iter = 0.0 : 0.05 : 0.2 
for i in noise_vec_iter
    for j = 1:5 
        push!(noise_vec, i)
    end
end 

noise = 0.01 


## ============================================ ##

# function sim_ode_test( fn, noise, stand_data_option ) 

    data_train, data_test = ode_train_test( fn, noise, stand_data_option ) 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

    # SINDy vs GPSINDy 
    λ = 0.1 
    
    # run SINDy on truth data  
    Ξ_true_stls       = sindy_stls( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
    Ξ_true_stls_terms = pretty_coeffs( Ξ_true_stls, data_train.x_true, data_train.u ) 
    
    # run SINDy (STLS, LASSO) and GPSINDy (LASSO) 
    Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms = gpsindy_Ξ_fn( data_train.t, data_train.x_true, data_train.dx_true, λ, data_train.u ) 

    # Train NN on the data
    # Define the 2-layer MLP
    dx_noise_nn = 0 * data_train.dx_noise 
    for i = 1 : x_vars 
        dx_noise_nn[:,i] = train_nn_predict(data_train.x_noise, data_train.dx_noise[:, i], 100, 2)
    end 

    # Concanate the two outputs to make a Matrix
    Ξ_nn_lasso  = sindy_lasso(x_train_noise, dx_noise_nn, λ)

    # ----------------------- # 
    # validate 
    
    # dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true_stls ) 
    dx_fn_sindy_stls  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
    dx_fn_sindy_lasso = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_lasso ) 
    dx_fn_nn_lasso    = build_dx_fn(poly_order, x_vars, u_vars, Ξ_nn_lasso)
    dx_fn_gpsindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    
    x0 = data_test.x_true[1,:] 
    # x_int_euler_true   = integrate_euler( dx_fn_true, x0, data_test.t, data_test.u ) 
    x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, data_test.t, data_test.u ) 
    x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, data_test.t, data_test.u ) 
    x_nn_lasso_test    = integrate_euler( dx_fn_nn_lasso, x0, data_test.t, data_test.u ) 
    x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u ) 
    
    t_test = data_test.t 
    
    # ----------------------- # 
    # plot smoothed data and validation test data 
    
    plot_validation_test( t_test, data_test.x_true, data_test.x_noise, x_sindy_stls_test, x_sindy_lasso_test, x_nn_lasso_test, x_gpsindy_test, noise ) 

# end 


## ============================================ ## 





