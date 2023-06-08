

## ============================================ ##
# split into training and validation data 

export split_train_test 
function split_train_test(x, train_fraction)

    ind = Int(round( size(x,1) * train_fraction ))  

    x_train = x[1:ind,:]
    x_test  = x[ind:end,:] 

    return x_train, x_test 

end 


## ============================================ ##
# plot state 

using Plots 

export plot_dyn 
function plot_dyn(t, x, str)

    n_vars = size(x,2) 

    # construct empty vector for plots 
    plot_vec_x = [] 
    for i = 1:n_vars 
        plt = plot(t, x[:,i], title = "State $(i)")
        push!(plot_vec_x, plt)
    end 
    plot_x = plot(plot_vec_x ..., 
        layout = (n_vars,1), 
        size = [600 n_vars*300], 
        xlabel = "Time (s)", 
        plot_title = "Dynamics. ODE fn = $( str )" ) 
    display(plot_x)      

    return plot_x 

end 


## ============================================ ##
# plot derivatives 

using Plots 

export plot_deriv
function plot_deriv(t, dx_true, dx_fd, dx_tv, str) 

    n_vars = size(dx_true, 2) 

    plot_vec_dx = [] 
    for j in 1 : n_vars
        plt = plot(t, dx_true[:,j], 
            title = "State $(j)", label = "true" ) 
            plot!(t, dx_fd[:,j], ls = :dash, label = "finite diff" )
            plot!(t, dx_tv[:,j], ls = :dash, label = "var diff" )
    push!( plot_vec_dx, plt ) 
    end

    plot_dx = plot(plot_vec_dx ... , 
        layout = (n_vars, 1), 
        size = [600 n_vars*300], 
        plot_title = "Derivatives. ODE fn = $( str )" )
    display(plot_dx) 

    return plot_dx 

end 


## ============================================ ## 
# plot validation data 

using Plots 

export validate_plot_data 
function validate_plot_data( t_train, x_train, t_test, x_test, z_fd, Ξ_fd, poly_order, plot_option )

    dx_gpsindy_fn = build_dx_fn(poly_order, z_fd) 
    dx_sindy_fn  = build_dx_fn(poly_order, Ξ_fd)

    t_gpsindy_val, x_gpsindy_val = validate_data(t_test, x_test, dx_gpsindy_fn)
    t_sindy_val, x_sindy_val     = validate_data(t_test, x_test, dx_sindy_fn)

    if plot_option == 1
        p = plot_validate(t_train, x_train, t_test, x_test, 
        t_gpsindy_val, x_gpsindy_val, t_sindy_val, x_sindy_val)
    end 

    return t_gpsindy_val, x_gpsindy_val, t_sindy_val, x_sindy_val, p 

end 


## ============================================ ##

export plot_validate 
function plot_validate(t_train, x_train, t_test, x_test, 
    t_gpsindy_val, x_gpsindy_val, t_sindy_val, x_sindy_val)

    n_vars = size(x_train,2) 

    plot_vec = [] 
    for i = 1:n_vars 

        # display training data 
        p = plot(t_train, x_train[:,i], 
            lw = 3, 
            c = :black, 
            label = "train (70%)", 
            grid = false, 
            xlim = (t_train[end]*3/4, t_test[end]), 
            legend = :outerright , 
            xlabel = "Time (s)", 
            title = "State $(i)", 
            ) 
        plot!(t_test, x_test[:,i], 
            ls = :dash, 
            c = :blue,
            lw = 3,  
            label = "test (30%)" 
            )
        plot!(t_gpsindy_val, x_gpsindy_val[:,i], 
            ls = :dashdot, 
            lw = 1.5, 
            c = :red, 
            label = "GP SINDy" 
            )
        plot!(t_sindy_val, x_sindy_val[:,i], 
            ls = :dot, 
            lw = 1.5, 
            c = :green, 
            label = "SINDy" 
            )

        push!( plot_vec, p ) 

    end 

    p_train_val = plot(plot_vec ... , 
        layout = (n_vars, 1), 
        size = [600 n_vars*300], 
        plot_title = "Training vs. Validation Data", 
        default(show = true) 
    )
    display(p_train_val) 
    
    # plot_dx = plot(plot_vec_dx ... , 
    #     layout = (n_vars, 1), 
    #     size = [600 n_vars*300], 
    #     plot_title = "Derivatives. ODE fn = $( str )" )
    # display(plot_dx) 

    return p_train_val 

end 


