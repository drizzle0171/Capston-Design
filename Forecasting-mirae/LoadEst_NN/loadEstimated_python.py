import numpy
import torch
import datetime
from load_estimation import *
import matplotlib.pyplot as plt
import random
    
nowDate = datetime.datetime(2023, 2, 1, 0)
dbClass_Seoultech_Future = DBSeoultech_LOADEST(
    timeInfo={
        "ST" : nowDate,
        "ED" : nowDate + datetime.timedelta(days=2)},
    connectInfo={
        "SERVER" : "114.71.51.11:21240",
        "DBNAME" : "서울과기대_PMS_log",
        "USER"   : "sa",
        "PWD"    : "Rceit1!"})
dbClass_Seoultech_Future.fetchDBTableData()

lstm = miraeload_HorizonEst('LSTM', "./LoadEst_NN/load_estimation/nn_model/Seq2Seq_final.pt", nowDate)
dlinear = miraeload_HorizonEst('DLinear', "./LoadEst_NN/load_estimation/nn_model/DLinear.pth", nowDate)
nlinear = miraeload_HorizonEst('NLinear', "./LoadEst_NN/load_estimation/nn_model/NLinear.pth", nowDate)
linear = miraeload_HorizonEst('Linear', "./LoadEst_NN/load_estimation/nn_model/Linear.pth", nowDate)
better_dlinear = miraeload_HorizonEst('betterDLinear', "./LoadEst_NN/load_estimation/nn_model/better_DLinear.pth", nowDate)
better_nlinear = miraeload_HorizonEst('betterNLinear', "./LoadEst_NN/load_estimation/nn_model/better_NLinear.pth", nowDate)
better_linear = miraeload_HorizonEst('betterLinear', "./LoadEst_NN/load_estimation/nn_model/better_Linear.pth", nowDate)

plt.figure(figsize=(18, 11))
dateAxis = dbClass_Seoultech_Future.getGridSampledProfile(3600)[0][:12]
plt.ylim([-50, 300])
plt.ylabel("kW", fontdict={"size":20})
plt.plot(dateAxis, lstm, linewidth= 2.0)
plt.plot(dateAxis, linear, linewidth= 2.0)
plt.plot(dateAxis, nlinear, linewidth= 2.0)
plt.plot(dateAxis, dlinear, linewidth= 2.0)
plt.plot(dateAxis, better_linear, linewidth= 2.0)
plt.plot(dateAxis, better_nlinear, linewidth= 2.0)
plt.plot(dateAxis, better_dlinear, linewidth= 2.0)
plt.plot(dbClass_Seoultech_Future.getGridSampledProfile(3600)[0][:12], dbClass_Seoultech_Future.getGridSampledProfile(3600)[1][:12], linewidth= 2.0)
plt.legend(["Estimated Profile by LSTM", "Estimated Profile by Linear", "Estimated Profile by NLinear", "Estimated Profile by DLinear", "Estimated Profile by Better-Linear", "Estimated Profile by Better-NLinear", "Estimated Profile by Better-DLinear", "Actual Sampled Load Data"], fontsize=14)
plt.grid()
plt.savefig('./visual/230201.png')
