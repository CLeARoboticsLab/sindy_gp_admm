using GaussianSINDy 
using LinearAlgebra 


## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

x_err_hist  = x_err_struct([], [], [], [])
for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 
end 


## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

x_err_hist  = x_err_struct([], [], [], [])
for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 
end 

























## ============================================ ## 
# single run (good) 


csv_file = "test/data/rollout_4_mod_u.csv"

Ξ_gpsindy = [] 
x_gpsindy = [] 
x_sindy   = [] 
t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 )

using Plots 
plot( x_test_noise[:,1], x_test_noise[:,2] ) 
plot!( x_test_sindy[:,1], x_test_sindy[:,2] ) 
plot!( x_test_gpsindy[:,1], x_test_gpsindy[:,2] ) 


## ============================================ ##
# save outputs as csv 
header = [ "t", "x1_test", "x2_test", "x3_test", "x4_test", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy" ] 

data   = [ t_test x_test_noise x_test_sindy x_test_gpsindy ]
df     = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single", ".csv"), df, header=header)


## ============================================ ##
data_noise = [ t_test x_test_noise ] 
header     = [ "t", "x1_test", "x2_test", "x3_test", "x4_test" ]
data       = [ t_test x_test_noise ]  
df         = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single_test_data", ".csv"), df, header=header)

