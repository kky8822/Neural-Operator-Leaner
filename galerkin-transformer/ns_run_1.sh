# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat 10 10 model_ns_V1000_Ffix_T1024_V200_10to10.pt result_ns_V1000_Ffix_T1024_V200_10to10.pt
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat 10 40 model_ns_V1000_Ffix_T1024_V200_10to40.pt result_ns_V1000_Ffix_T1024_V200_10to40.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat 10 10 model_ns_V10000_Ffix_T1024_V200_10to10.pt result_ns_V10000_Ffix_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat 10 40 model_ns_V10000_Ffix_T1024_V200_10to40.pt result_ns_V10000_Ffix_T1024_V200_10to40.pt cuda:1 False False

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat 10 10 model_ns_V1000_Fvar_T1024_V200_10to10.pt result_ns_V1000_Fvar_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat 10 40 model_ns_V1000_Fvar_T1024_V200_10to40.pt result_ns_V1000_Fvar_T1024_V200_10to40.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 10 model_ns_V10000_Fvar_T1024_V200_10to10.pt result_ns_V10000_Fvar_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 40 model_ns_V10000_Fvar_T1024_V200_10to40.pt result_ns_V10000_Fvar_T1024_V200_10to40.pt cuda:0

# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat 10 10 model_ns_Vmix_Fvar_T1024_V200_10to10.pt result_ns_Vmix_Fvar_T1024_V200_10to10.pt cuda:1 False False
# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat 10 40 model_ns_Vmix_Fvar_T1024_V200_10to40.pt result_ns_Vmix_Fvar_T1024_V200_10to40.pt cuda:1 False False

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat        10 10 rk2 model_ns_V1000_Ffix_T1024_V200_10to10_rk2.pt  result_ns_V1000_Ffix_T1024_V200_10to10_rk2.pt  cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat       10 10 rk2 model_ns_V10000_Ffix_T1024_V200_10to10_rk2.pt result_ns_V10000_Ffix_T1024_V200_10to10_rk2.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat  10 10 rk2 model_ns_V1000_Fvar_T1024_V200_10to10_rk2.pt  result_ns_V1000_Fvar_T1024_V200_10to10_rk2.pt  cuda:1
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 10 rk2 model_ns_V10000_Fvar_T1024_V200_10to10_rk2.pt result_ns_V10000_Fvar_T1024_V200_10to10_rk2.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat   10 10 rk2 model_ns_Vmix_Fvar_T1024_V200_10to10_rk2.pt   result_ns_Vmix_Fvar_T1024_V200_10to10_rk2.pt   cuda:1

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat        10 40 rk2 model_ns_V1000_Ffix_T1024_V200_10to40_rk2.pt  result_ns_V1000_Ffix_T1024_V200_10to40_rk2.pt  cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat       10 40 rk2 model_ns_V10000_Ffix_T1024_V200_10to40_rk2.pt result_ns_V10000_Ffix_T1024_V200_10to40_rk2.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat  10 40 rk2 model_ns_V1000_Fvar_T1024_V200_10to40_rk2.pt  result_ns_V1000_Fvar_T1024_V200_10to40_rk2.pt  cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 40 rk2 model_ns_V10000_Fvar_T1024_V200_10to40_rk2.pt result_ns_V10000_Fvar_T1024_V200_10to40_rk2.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat   10 40 rk2 model_ns_Vmix_Fvar_T1024_V200_10to40_rk2.pt   result_ns_Vmix_Fvar_T1024_V200_10to40_rk2.pt   cuda:1

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat        10 10 euler True model_ns_V1000_Ffix_T1024_V200_10to10_ST_euler.pt  result_ns_V1000_Ffix_T1024_V200_10to10_ST_euler.pt  cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat       10 10 euler True model_ns_V10000_Ffix_T1024_V200_10to10_ST_euler.pt result_ns_V10000_Ffix_T1024_V200_10to10_ST_euler.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat  10 10 euler True model_ns_V1000_Fvar_T1024_V200_10to10_ST_euler.pt  result_ns_V1000_Fvar_T1024_V200_10to10_ST_euler.pt  cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 10 euler True model_ns_V10000_Fvar_T1024_V200_10to10_ST_euler.pt result_ns_V10000_Fvar_T1024_V200_10to10_ST_euler.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat   10 10 euler True model_ns_Vmix_Fvar_T1024_V200_10to10_ST_euler.pt   result_ns_Vmix_Fvar_T1024_V200_10to10_ST_euler.pt   cuda:1

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat 10 10 euler model_GT3d_ns_V1000_Ffix_T1024_V200_10to10.pt result_GT3d_ns_V1000_Ffix_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50.mat 10 40 euler model_GT3d_ns_V1000_Ffix_T1024_V200_10to40.pt result_GT3d_ns_V1000_Ffix_T1024_V200_10to40.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat 10 10 euler model_GT3d_ns_V10000_Ffix_T1024_V200_10to10.pt result_GT3d_ns_V10000_Ffix_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50.mat 10 40 euler model_GT3d_ns_V10000_Ffix_T1024_V200_10to40.pt result_GT3d_ns_V10000_Ffix_T1024_V200_10to40.pt cuda:0

# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat 10 10 euler model_GT3d_ns_V1000_Fvar_T1024_V200_10to10.pt result_GT3d_ns_V1000_Fvar_T1024_V200_10to10.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_V1000_N5000_T50_var_f.mat 10 40 euler model_GT3d_ns_V1000_Fvar_T1024_V200_10to40.pt result_GT3d_ns_V1000_Fvar_T1024_V200_10to40.pt cuda:1
# python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 10 euler model_GT3d_ns_V10000_Fvar_T1024_V200_10to10.pt result_GT3d_ns_V10000_Fvar_T1024_V200_10to10.pt cuda:1
python ex4_navier_stokes_2+1d.py ns_V10000_N5000_T50_var_f.mat 10 40 euler model_GT3d_ns_V10000_Fvar_T1024_V200_10to40.pt result_GT3d_ns_V10000_Fvar_T1024_V200_10to40.pt cuda:1

# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat 10 10 euler model_GT3d_ns_Vmix_Fvar_T1024_V200_10to10.pt result_GT3d_ns_Vmix_Fvar_T1024_V200_10to10.pt cuda:0
# python ex4_navier_stokes_2+1d.py ns_Vmix_N3000_T50_var_f.mat 10 40 euler model_GT3d_ns_Vmix_Fvar_T1024_V200_10to40.pt result_GT3d_ns_Vmix_Fvar_T1024_V200_10to40.pt cuda:0