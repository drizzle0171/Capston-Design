import numpy
import datetime
from pytimekr import pytimekr

class DateEncode_onehot:
    @staticmethod
    def onehotEncode(idx:int, len:int):
        vect = numpy.zeros((len), dtype=int)
        vect[idx - 1] = int(1)
        return vect
    @staticmethod
    def hrIdx(hour:int)->numpy.ndarray:
        match(hour):
            case 0|1|2:
                idx = 1
                pass
            case 3|4|5:
                idx = 2
                pass
            case 6|7|8:
                idx = 3
                pass
            case 9|10|11:
                idx = 4
                pass
            case 12|13|14:
                idx = 5
                pass
            case 15|16|17:
                idx = 6
                pass
            case 18|19|20:
                idx = 7
                pass
            case 21|22|23:
                idx = 8
                pass
        return DateEncode_onehot.onehotEncode(idx, 8).reshape(8, 1)
    @staticmethod
    def wkday(day:int)->numpy.ndarray:
        return DateEncode_onehot.onehotEncode(day + 1, 7).reshape(7, 1)
    @staticmethod
    def season(month:int)->numpy.ndarray:
        match(month):
            case 11|12|1|2:
                idx = 1
                pass
            case 3|4|5:
                idx = 2
                pass
            case 6|7|8|9:
                idx = 3
                pass
            case 10:
                idx = 4
                pass
        return DateEncode_onehot.onehotEncode(idx, 4).reshape(4, 1)
    @staticmethod
    def holidy(time:datetime.datetime)->numpy.ndarray:
        list = pytimekr.holidays()
        time = time.date() 
        if time.weekday()==6:
            holiday_yn=1
        elif time in list:
            holiday_yn=1
        else:
            holiday_yn=2
        return DateEncode_onehot.onehotEncode(holiday_yn, 2).reshape(2, 1)

def datetime_onehotEncode(date:datetime.datetime)->numpy.ndarray:
    return numpy.vstack((
        DateEncode_onehot.hrIdx(date.hour),
        DateEncode_onehot.wkday(date.weekday()),
        DateEncode_onehot.season(date.month),
        DateEncode_onehot.holidy(date)
    ))
def datetime_72HrEncode(edPoint:datetime.datetime):
    encoderList = list([None for _ in range(0, 72)])
    edPoint_Clean = edPoint.replace(microsecond=0)
    for x in range(0, 72):
        posDay = edPoint_Clean + datetime.timedelta(hours=(x - 71))
        encoderList[x] = datetime_onehotEncode(posDay).T
    return numpy.vstack(encoderList)

def datetime_for_better(date:datetime.datetime)->numpy.ndarray:
    return numpy.vstack((date.month, date.day, date.weekday(), date.hour))

def datetime_72HrEncode_better(edPoint:datetime.datetime):
    encoderList = list([None for _ in range(0, 72)])
    edPoint_Clean = edPoint.replace(microsecond=0)
    for x in range(0, 72):
        posDay = edPoint_Clean + datetime.timedelta(hours=(x - 71))
        encoderList[x] = datetime_for_better(posDay).T
    return numpy.vstack(encoderList)