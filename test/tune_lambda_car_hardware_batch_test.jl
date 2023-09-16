using GaussianSINDy 
using LinearAlgebra 

# ----------------------- # 
# batch run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
# for i in eachindex(csv_files_vec)  
for i = [1, 2]
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

x_err_hist  = x_err_struct([], [], [], [])
for i = eachindex(csv_files_vec) 
# for i = [ 4, 5 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 
end 

# compute average error and variance 
mean_gpsindy_err = mean( x_err_hist.gpsindy ) 
mean_sindy_err   = mean( x_err_hist.sindy_lasso ) 

var_gpsindy_err  = var( x_err_hist.gpsindy ) 
var_sindy_err    = var( x_err_hist.sindy_lasso ) 

println( "mean of mean of gpsindy error: ", mean(mean_gpsindy_err) ) 
println( "mean of mean of sindy error: ", mean(mean_sindy_err) )

println( "mean of var of gpsindy error: ", mean(var_gpsindy_err) ) 
println( "mean of var of sindy error: ", mean(var_sindy_err) ) 


