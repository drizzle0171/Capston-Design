
from croquette_box.simul_dataset.database_interface import DataBaseInterface
from croquette_box.simul_dataset import DataBaseTable, process_rawTableData
import numpy, datetime

class DBSeoultech_LOADEST(DataBaseInterface):
    def __init__(self, timeInfo: dict, connectInfo: dict) -> None:
        super().__init__(timeInfo, [3, 4, 5, 10, 11], connectInfo)
        self.addConnectInfo(connectInfo)
        return
    def addConnectInfo(self, connectInfo) -> None:
        self.__server = self.connectInfo["SERVER"]
        self.__dbname = self.connectInfo["DBNAME"]
        self.__user   = self.connectInfo["USER"]
        self.__pwd    = self.connectInfo["PWD"]
        return
    def connectDataBase(self) -> None:
        import pymssql
        self.__cnxn =  pymssql.connect(
            self.__server, self.__user, self.__pwd, self.__dbname)
        self.__sqlcursor = self.__cnxn.cursor()
        return None
    def disconnectDataBase(self)->None:
        self.__cnxn.close()
        return
    def fetchDBTableData(self) -> None:
        self.__table = list([])
        self.connectDataBase()
        if(self.timeInfo["ST"] != None and
           self.timeInfo["ED"] != None):
            for x in self.tableIndex:
                try:
                    target="[서울과기대_PMS_log].[dbo].[SCADA_HISTORY_DATA2_{}] ".format(x)
                    self.__sqlcursor.execute(
                        "SELECT * "+
                        "FROM {}".format(target)+
                        "WHERE savingtime BETWEEN \'"+
                        "{}-{}-{} {}".format(
                            self.timeInfo["ST"].month,
                            self.timeInfo["ST"].day,
                            self.timeInfo["ST"].year,
                            self.timeInfo["ST"].time())+
                        "\' AND \'"+
                        "{}-{}-{} {}".format(
                            self.timeInfo["ED"].month,
                            self.timeInfo["ED"].day,
                            self.timeInfo["ED"].year,
                            self.timeInfo["ED"].time())+"\'")
                    self.__table.append(numpy.array(self.__sqlcursor.fetchall()))
                    self.__table[-1] = DataBaseTable(process_rawTableData(self.__table[-1]))
                except:
                    self.__table.append(None)
        self.disconnectDataBase()
        self.saveDBData()
        return None
    def getDBTableData(self)->list:
        return self.__table
    def getIntvData(self) -> None:
        return super().getIntvData()
    def saveDBData(self)->None:
        self.__tProfile  = self.getDBTableData()[4].selectRow(num=11, div=1e3)[0]
        self.__gridData  = self.getDBTableData()[4].selectRow(num=11, div=1e3)[1]
        self.__essPData  = sum([self.getDBTableData()[x].selectRow(num=4, div=1.0)[1] for x in range(0, 3)])
        self.__essNData  = numpy.zeros(self.__essPData.shape)
        self.__pvGenData = self.getDBTableData()[3].selectRow(num=2)[1]
        return
    def getProfileTime(self)->numpy.ndarray:
        return self.__tProfile
    def getLoadProfile(self)->numpy.ndarray:
        essPWRData = self.__essPData - self.__essNData
        return self.__gridData - essPWRData + self.__pvGenData
    def getGridProfile(self)->numpy.ndarray:
        return self.__gridData
    def getPVGenProfile(self)->numpy.ndarray:
        return self.__pvGenData
    def getESSPWRProfile(self)->numpy.ndarray:
        return self.__essPData - self.__essNData
    def getLoadSampledProfile(self, samp:float)->tuple((list, numpy.ndarray)):
        essPWRData_P = sum([self.getDBTableData()[x].selectRow_sample(sampSec=samp, num=4, div=1.0)[1] for x in range(0, 3)])
        essPWRData_N = numpy.zeros(essPWRData_P.shape)
        gridData  = self.getDBTableData()[4].selectRow_sample(sampSec=samp, num=11, div=1e3)
        pvGenData = self.getDBTableData()[3].selectRow_sample(sampSec=samp, num=2, div=1.0)[1]
        return gridData[0], gridData[1] - (essPWRData_P - essPWRData_N) + pvGenData
    def getGridSampledProfile(self, samp:float)->tuple((list, numpy.ndarray)):
        gridData  = self.getDBTableData()[4].selectRow_sample(sampSec=samp, num=11, div=1e3)
        return gridData
    def getPVGenSampledProfile(self, samp:float)->tuple((list, numpy.ndarray)):
        pvGenData = self.getDBTableData()[3].selectRow_sample(sampSec=samp, num=2, div=1.0)
        return pvGenData
    def findTimeIndex(self, idxtime:datetime.datetime)->int:
        return numpy.where(self.__tProfile == idxtime)[0]
    def pointMeasure(self, idxtime:datetime.datetime)->dict:
        idx = self.findTimeIndex(idxtime)
        return dict(
        {
            "PV"   : self.getPVGenProfile()[idx],
            "GRID" : self.getGridProfile()[idx],
            "LOAD" : self.getLoadProfile()[idx],
            "PESS" : self.getESSPWRProfile()[idx]
        })