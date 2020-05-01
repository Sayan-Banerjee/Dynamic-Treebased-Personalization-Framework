import numpy as np
import pandas as pd
import re
import logging

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handleMissingValues(df, tolerance = 0.90):    
    initialSize = df.shape[0] * df.shape[1]
    while max(df.isnull().sum(axis=0))!=0 > 0.05 * df.shape[0]: # More than 5% blank/Null for a column
        colWise = dict(df.isnull().sum(axis=0))
        colToBeDropped = max(colWise, key = colWise.get)
        df.drop(colToBeDropped, axis=1, inplace = True)
    df.dropna(axis=0, inplace=True)
    finalSize = df.shape[0] * df.shape[1] 
    if finalSize <= initialSize*tolerance:
        logger.error("Too much missing data from the orginal source file!")
        raise ValueError("Too much missing data from the orginal source file!")
        
    return df

def weekendCalc(x):
    j = re.findall("(friday|saturday|fri|sat)", x)
    if j:
        return 1
    else:
        return 0

def midWeekCalc(x):
    j = re.findall("(tuesday|wednesday|thursday|tue|wed|thu)", x)
    if j:
        return 1
    else:
        return 0
    
def earlyWeekCalc(x):
    j = re.findall("(sunday|monday|sun|mon)", x)
    if j:
        return 1
    else:
        return 0

def calcWeekEndMidEarlyWeek(df, keyString="arrival"):
    
    columnsSet = list(set(df.columns))
    df = df[columnsSet].copy(deep=True)
    weekdayFlag = [0 for i in columnsSet]
    j = 0
    for i in columnsSet:
        x = re.findall("week", i)
        if x:
            weekdayFlag[j] = 1
        j+=1

    j = 0
    for i in columnsSet:
        x = re.findall("day", i)
        if not x:
            weekdayFlag[j] = 0
        j+=1
    weekDayCols = []    
    indices = [i for i, x in enumerate(weekdayFlag) if x == 1]
    for idx in indices:
        col = columnsSet[idx]
        weekDayCols.append(col)
        df['{}_{}'.format(col, 'weekend')] = df[col].apply(lambda x: weekendCalc(x))
        df['{}_{}'.format(col, 'midweek')] = df[col].apply(lambda x: midWeekCalc(x))
        df['{}_{}'.format(col, 'earlyweek')] = df[col].apply(lambda x: earlyWeekCalc(x))
    
    dateCols =[]
    totalDataPoints = len(df)
    for i in columnsSet:
        flag = True
        indx = np.random.randint(low=0, high=totalDataPoints, size=3)
        for ii in indx:
            val = df.loc[ii, i]
            if isinstance(val, str) and not val.replace('.', '', 1).lstrip('-').isdigit():
                try:
                    pd.to_datetime(val)
                except:
                    flag = False
                    break
            else:
                flag=False
                break
        if flag:
            dateCols.append(i)
        
    keyStringDateCol = None
    keyStringMonthCol = None
    keyStringWeekdayCol = None

    if keyString is not None:
        for c in dateCols:
            if re.findall(keyString, c):
                keyStringDateCol = c
                break

        for c in columnsSet:
            if keyString is not None and re.findall(keyString, c) and re.findall("month", c):
                keyStringMonthCol=c
                break

        for c in weekDayCols:
            if keyString is not None and re.findall(keyString, c):
                keyStringWeekdayCol = c
            
    return df, dateCols, keyStringDateCol, weekDayCols, keyStringWeekdayCol, keyStringMonthCol

def getData(sourcePath, categoricalAttributes, keyString="arrival", conquerDataColumns=None, tolerance=0.90, delimiter=';'):
    #  Sample Query From Vertica(ADS):
    #      select c.CAL_DT as ArrivalDate,
    #      TO_CHAR(c.CAL_DT, 'DAY') as ArrivalDate_Weekday,MONTH(c.CAL_DT) as Arrival_Month,
    #      a.channel_id, (c.CAL_DT::date - a.crs_confirm_dt::date) as LeadDays, a.lang_id, a.nights_qty, a.room_qty
    #      from SHSCRSRepl_nonpd.REZ a
    #      inner join SHSCRSRepl_nonpd.Hotel b
    #      on a.HOTEL_GUID = b.HOTEL_GUID
    #      inner join SHSCRSRepl_nonpd.CAL c
    #      on a.ARRIVAL_CAL_ID = c.CAL_ID
    #      where b.Hotel_ID = 59053
    #      and c.CAL_DT >= ADD_MONTHS(CURRENT_DATE, -12)
    #      and c.CAL_DT < CURRENT_DATE
    #      and rez_status_id in (5,9);

    try:
        df = pd.read_csv(sourcePath, delimiter=delimiter)  # Please make sure the source file is delimited by ';'
    except IOError:
        logger.error ("Could not read the file at {}".format(sourcePath))
        raise IOError ("Could not read the file at {}".format(sourcePath))
        return None
    
    for col in df.columns:
        if col == 'Unnamed: 0' or col == 'unnamed: 0':
            df.drop(columns=col, inplace=True)
        elif isinstance(df[col][0], str):
            df[col] = df[col].apply(lambda x: x.strip().lower())

    if keyString is not None and len(str(keyString).strip()) > 0:
        keyString = str(keyString).lower().strip()
    else:
        keyString = None

    if conquerDataColumns is not None:
        if isinstance(conquerDataColumns, list) or isinstance(conquerDataColumns, tuple) or \
                isinstance(conquerDataColumns, set):
            conquerDataColumns = [cdc.strip().lower() for cdc in conquerDataColumns]
        elif isinstance(conquerDataColumns, str):
            conquerDataColumns = conquerDataColumns.strip().lower()
            conquerDataColumns = [conquerDataColumns]
        else:
            conquerDataColumns = [conquerDataColumns]

        values = dict()
        for i in conquerDataColumns:
            values[i] = 0.
        df.fillna(value=values, inplace=True)

    df = handleMissingValues(df, tolerance)
    
    newColumnNames = dict()
    for col in df.columns:
        newColumnNames[col] = col.strip().lower()
        
    df.rename(columns=newColumnNames, inplace=True)
    
    defaultValues = dict(df.mode().loc[0])

    if conquerDataColumns is not None:
        defaultValues = {k: v for k, v in defaultValues.items() if k not in conquerDataColumns}
                         
    df, dateCols, keyStringDateCol, weekDayCols, keyStringWeekdayCol, keyStringMonthCol =\
                                            calcWeekEndMidEarlyWeek(df, keyString)
    
    logger.info("All columns after feature engineering and before dropping any: {}".format(list(df.columns)))
    
    if categoricalAttributes is not None:
        weekDayCols = list(set(weekDayCols) - set(categoricalAttributes))
        
    df.drop(columns=weekDayCols, inplace = True)
    
    if dateCols is not None and len(dateCols) > 0:
        dateCols = list(set(dateCols) - set(keyStringDateCol))
        df.drop(columns=dateCols, inplace = True)    
    
    # need to verify if the columns exist in df
    # need to verify none of them mentioned in categorical variable names.
    # need to verify if all are numerical columns
    set_df_cols = set(list(df.columns))
    set_cqr_cols = set()
    if conquerDataColumns is not None:
        set_cqr_cols = set(conquerDataColumns)
        assert len(set_cqr_cols - set_df_cols) == 0, "Some of the columns mentioned in 'conquerDataColumns', do NOT exist in the provided data!"
    if conquerDataColumns is not None and categoricalAttributes is not None:
        assert len(set_cqr_cols.intersection(set(categoricalAttributes))) == 0, "Current version only supports numerical type conquer data!"
    if conquerDataColumns is not None:
        for col in conquerDataColumns:
            for ii in range(len(df)):
                val = df.loc[ii, col]
                if isinstance(val, str) and not val.replace('.', '', 1).lstrip('-').isdigit():
                    logger.error("Current version only supports numerical type conquer data! Error occurred for value: {} at column: {} at row: {}".format(val, col, ii))
                    raise ValueError("Current version only supports numerical type conquer data! Error occurred for value: {} at column: {} at row: {}".format(val, col, ii))
                    
    return df, keyStringDateCol, keyStringWeekdayCol, keyStringMonthCol, defaultValues


def calculateOtherGroupMember(df, attribute, min_split):
    individual = []
    others = []
    tmp = df[attribute].value_counts()
    total = tmp.sum()
    lowerLimit = total * min_split
    for i in list(tmp.index):
        if tmp.loc[i] < lowerLimit:
            others.append(i)
        else:
            individual.append(i)        
    return individual, others


def getOthersGoup(data, categoricalAttributes, min_split = 0.10):
    othersGroup = {}
    if not isinstance(categoricalAttributes, list):
        categoricalAttributes = [categoricalAttributes]
    
    for attribute in categoricalAttributes:
        individual, others = calculateOtherGroupMember(data, attribute, min_split)
        if len(individual) == 0:
            data.drop(columns=attribute, inplace = True)
            continue
        
        for item in individual:
            data["{}_{}".format(attribute, item)] = data[attribute].apply(lambda x: 1 if x==item else 0)
        if len(others) > 0:
            data["{}_{}".format(attribute, 'others')] = data[attribute].apply(lambda x: 1 if x in others else 0)
            othersGroup[attribute] = others
            
        data.drop(columns=attribute, inplace = True)

    return othersGroup, data

