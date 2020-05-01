import pandas as pd
import numpy as np
import logging

logging.basicConfig(format="%(asctime)s - %(thread)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def returnBucket(x):
    if x > 120:
        return '120+'
    elif x > 90:
        return '91-120'
    elif x > 60:
        return '90-61'
    elif x > 45:
        return '60-46'
    elif x > 30:
        return '45-31'
    elif x > 21:
        return '30-22'
    elif x > 14:
        return '21-15'
    elif x > 10:
        return '14-11'
    elif x > 7:
        return '10-8'
    elif x > 4:
        return '7-5'
    elif x > 2:
        return '4-3'
    elif x > 0:
        return '2-1'
    else:
        return '0'


def bucketizeLeadDays(df, leadDaysColName='leaddays'):
    bucketColName = 'LeadDays_Bucket'
    assert leadDaysColName in df.columns, "The column name: {} Not Found in the passed DataFrame".format(
        leadDaysColName)

    df[bucketColName] = df[leadDaysColName].apply(lambda x: returnBucket(x))
    _map = {0: '120+', 1: '91-120', 2: '90-61', 3: '60-46', 4: '45-31', 5: '30-22',
            6: '21-15', 7: '14-11', 8: '10-8', 9: '7-5', 10: '4-3', 11: '2-1', 12: '0'}
    return df, _map, bucketColName


def _getUnitVector(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)

    magnitude = np.sqrt(np.sum(np.square(vector)))
    vector = vector / magnitude
    return vector


def monthToMonthCosSim(record):
    # Each row of record is bucketed booking trend of each month
    # There is 12 months data. I am not putting validation in, because without 12 months
    # data, detecting seasonality doesn't make sense. If you do not have 12 months record
    # you shouldn't call this function.
    record2 = pd.DataFrame(columns=[i + 1 for i in range(12)], index=[i + 1 for i in range(12)])
    for i in range(12):
        month1 = i + 1
        a = list(record.loc[month1])
        a = _getUnitVector(a)
        for j in range(12):
            month2 = j + 1
            b = list(record.loc[month2])
            b = _getUnitVector(b)
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            record2.loc[month1, month2] = cos_sim

    return record2


def dataPrepForClusteringByBookingTrend(df, _map, keyStringDateCol, keyStringWeekdayCol,
                                        keyStringMonthCol, bucketColName='LeadDays_Bucket',
                                        groupByHowDissimilarToOthersTo=False):
    providedCols = list(df.columns)
    sumVsValueCounts_flag = False  # nights_qty and room_qty is not provided, will go for value counts by default.
    # value counts just mean, every reservation is considered for 1 night and 1 room.
    toBeFetchedColumns = [bucketColName]
    if 'nights_qty' in providedCols and 'room_qty' in providedCols:
        sumVsValueCounts_flag = True
        toBeFetchedColumns.append('nights_qty')
        toBeFetchedColumns.append('room_qty')

    if keyStringDateCol is None:  # Check if individual dates are not provided
        avgVsTotal_flag = False  # If so, we have no way to take average we have to go sum/total
    else:
        toBeFetchedColumns.insert(0, keyStringDateCol)
        avgVsTotal_flag = True  # As individual dates are provided we can calculate the average performace of
        # each criteria [early-week, mid-week, week-end] for each month.

    record = pd.DataFrame(columns=[i for i in range(3 * (len(_map)))], index=[i + 1 for i in range(12)])

    for i in range(12):
        month = i + 1

        tmp_0 = pd.DataFrame(df[(df[keyStringMonthCol] == month) &
                                (df['{}_earlyweek'.format(keyStringWeekdayCol)] == 1)])[toBeFetchedColumns]

        if sumVsValueCounts_flag:
            tmp_0['val'] = tmp_0['nights_qty'] * tmp_0['room_qty']
            tmp_0.drop(columns=['nights_qty', 'room_qty'], inplace=True)

        if avgVsTotal_flag:
            count = tmp_0[keyStringDateCol].nunique()
            tmp_0.drop(columns=[keyStringDateCol], inplace=True)
        else:
            count = 1.

        if sumVsValueCounts_flag:
            tmp_0 = tmp_0.groupby(by=bucketColName).sum()
        else:
            tmp_0 = tmp_0[bucketColName].value_counts()

        tmp_0 = tmp_0 / count

        tmp_1 = pd.DataFrame(df[(df[keyStringMonthCol] == month) &
                                (df['{}_midweek'.format(keyStringWeekdayCol)] == 1)])[toBeFetchedColumns]
        if sumVsValueCounts_flag:
            tmp_1['val'] = tmp_1['nights_qty'] * tmp_1['room_qty']
            tmp_1.drop(columns=['nights_qty', 'room_qty'], inplace=True)

        if avgVsTotal_flag:
            count = tmp_1[keyStringDateCol].nunique()
            tmp_1.drop(columns=[keyStringDateCol], inplace=True)
        else:
            count = 1.

        if sumVsValueCounts_flag:
            tmp_1 = tmp_1.groupby(by='LeadDays_Bucket').sum()
        else:
            tmp_1 = tmp_1['LeadDays_Bucket'].value_counts()
        tmp_1 = tmp_1 / count

        tmp_2 = pd.DataFrame(df[(df[keyStringMonthCol] == month) &
                                (df['{}_weekend'.format(keyStringWeekdayCol)] == 1)])[toBeFetchedColumns]
        if sumVsValueCounts_flag:
            tmp_2['val'] = tmp_2['nights_qty'] * tmp_2['room_qty']
            tmp_2.drop(columns=['nights_qty', 'room_qty'], inplace=True)

        if avgVsTotal_flag:
            count = tmp_2[keyStringDateCol].nunique()
            tmp_2.drop(columns=[keyStringDateCol], inplace=True)
        else:
            count = 1.

        if sumVsValueCounts_flag:
            tmp_2 = tmp_2.groupby(by='LeadDays_Bucket').sum()
        else:
            tmp_2 = tmp_2['LeadDays_Bucket'].value_counts()

        tmp_2 = tmp_2 / count

        val_0 = np.zeros(len(_map))
        val_1 = np.zeros(len(_map))
        val_2 = np.zeros(len(_map))
        for i in range(len(_map)):
            v0 = tmp_0.loc[_map.get(i)] if _map.get(i) in tmp_0.index else 0.
            v1 = tmp_1.loc[_map.get(i)] if _map.get(i) in tmp_1.index else 0.
            v2 = tmp_2.loc[_map.get(i)] if _map.get(i) in tmp_2.index else 0.
            val_0[i] = v0
            val_1[i] = v1
            val_2[i] = v2
            val = [val for pair in zip(val_0, val_1, val_2) for val in pair]

        record.loc[month] = val

    for col in record.columns:
        record[col] = ((record[col] - record[col].min()) / (record[col].max() - record[col].min()))

    if groupByHowDissimilarToOthersTo:
        record = monthToMonthCosSim(record)

    return record


def readSeparateSeasonalityDetectionData(sourcePath, delim=';'):
    #    *** This is a Placeholder Function to demostrate the future scope and purpose ***
    #    *** As it is a place holder function the data preparation and validations are not extensive ***
    #    At this stage the demo data(structured in .csv format) it reads, has each column as a attribute/product
    #    and each row potrays the the products/attributes shopped/purchased by each month for an entity(store).
    #    each index depicts numerical value of each month.
    try:
        df = pd.read_csv(sourcePath, delimiter=delim)
    except IOError:
        logger.error("Could not read the file at {}".format(sourcePath))
        raise IOError("Could not read the file at {}".format(sourcePath))
        return None

    newColumnNames = {}
    for col in df.columns:
        if col == 'Unnamed: 0' or col == 'unnamed: 0':
            df.drop(columns=col, inplace=True)
        else:
            newColumnNames[col] = col.strip()

    df.rename(columns=newColumnNames, inplace=True)
    df.fillna(value=0.0)
    df.reset_index(inplace=True, drop=True)
    df.index = df.index + 1
    return df
