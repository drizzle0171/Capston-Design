
import numpy
import torch
import datetime
import NLinear, Linear, DLinear, model
from .date_parser import datetime_72HrEncode, datetime_72HrEncode_better
from .seoultechDatabase  import DBSeoultech_LOADEST

def generate_InputFeature(pwrHist:numpy.ndarray, dateEncode:numpy.ndarray) -> numpy.ndarray:
    pwrHistory = numpy.array(pwrHist).reshape(72, 1) * 1000 / 250553.62
    inputFeat = numpy.hstack([pwrHistory, dateEncode])
    return numpy.array([inputFeat])

def predict_LoadHorizon(model_name, Model, inputFeature:numpy.ndarray, time=None) -> numpy.ndarray:
    if 'Linear' in model_name:
        predEstimation = Model(torch.tensor(inputFeature, dtype=torch.float), time).detach().numpy()
    else:
        predEstimation = Model.predict(torch.tensor(inputFeature, dtype=torch.float),12).detach().numpy()
    predEstimation = predEstimation[0,:,0] * 250.55362
    return predEstimation

def miraeload_HorizonEst(model_name, modelPath:str, nowDate : datetime.datetime):
    dbClass_Seoultech = DBSeoultech_LOADEST(
        timeInfo={
            "ST" : nowDate +  datetime.timedelta(hours=-72),
            "ED" : nowDate},
        connectInfo={
            "SERVER" : "114.71.51.11:21240",
            "DBNAME" : "서울과기대_PMS_log",
            "USER"   : "sa",
            "PWD"    : "Rceit1!"})
    dbClass_Seoultech.fetchDBTableData()
    pwrHistory_72Hr  = torch.Tensor(dbClass_Seoultech.getLoadSampledProfile(3600)[1] / 250.55362).unsqueeze(0).unsqueeze(2)
    
    if 'Linear' in model_name:
        if model_name == 'Linear':
            Model = Linear.Linear(72, 12)
        elif model_name == 'NLinear':
            Model = NLinear.NLinear(72, 12)
        elif model_name == 'DLinear':
            Model = DLinear.DLinear(72, 12)
        elif model_name == 'betterLinear':
            Model = Linear.betterLinear(72, 12)
        elif model_name == 'betterNLinear':
            Model = NLinear.betterNLinear(72, 12)
        elif model_name == 'betterDLinear':
            Model = DLinear.betterDLinear(72, 12)
        Model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')), strict=False)
        timeFeature = datetime_72HrEncode_better(nowDate)
        timeFeature = torch.Tensor(timeFeature).unsqueeze(0)
        estimated_Profile = predict_LoadHorizon(model_name, Model, pwrHistory_72Hr, timeFeature)
    else:
        seq2SeqModel = torch.load(modelPath, map_location= torch.device("cpu"))
        dateEncoded_72Hr = datetime_72HrEncode(nowDate)
        inputFeature_Gen = generate_InputFeature(pwrHistory_72Hr, dateEncoded_72Hr)
        estimated_Profile = predict_LoadHorizon(model_name, seq2SeqModel, inputFeature_Gen)

    return estimated_Profile