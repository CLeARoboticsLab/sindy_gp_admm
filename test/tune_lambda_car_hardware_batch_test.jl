using GaussianSINDy 
using LinearAlgebra 

# ----------------------- # 
# batch run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
# for i = [1, 2]
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

x_hist      = x_struct( [], [], [], [], [], [] )
x_err_hist  = x_err_struct([], [], [], [])
# for i = eachindex(csv_files_vec) 
for i = [1, 2, 3, 5, 6, 7, 8, 9, 10] 
# i = 4 
    println( "i = ", i ) 

    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    println( "i = ", i ) 

    # x hist 
    push!( x_hist.truth,       x_test_noise ) 
    push!( x_hist.sindy_lasso, x_test_sindy )
    push!( x_hist.gpsindy,     x_test_gpsindy ) 

    # error diagnostics 
    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 

    ## ============================================ ##
    # save outputs as csv 
    header = [ "t", "x1_test", "x2_test", "x3_test", "x4_test", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy" ] 

    data   = [ t_test x_test_noise x_test_sindy x_test_gpsindy ]
    df     = DataFrame( data,  :auto ) 
    CSV.write(string("car_hw", i, ".csv"), df, header=header)


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


