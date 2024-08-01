import pandas as pd

# BPD                   : 2022-11-16T11:00:00 - 2024-05-14T09:15:00
# sdoml-lite-biosentinel: 2022-11-01T00:00:00 - 2024-05-14T19:30:00
# CRaTER-D1D2           : 2009-06-26T21:00:00 - 2024-06-19T00:00:00

# Data taken from page: 
# Solar Proton Events Affecting the Earth Environment
# https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html

# Begin Time Yr M/D (UTC); Maximum Time Yr M/D (UTC); >10 MeV Maximum (pfu)
events_biosentinel = []
events_biosentinel.append(('2023-02-25T21:10:00', '2023-02-26T04:40', 58))
events_biosentinel.append(('2023-03-13T07:35:00', '2023-03-15T04:25', 22))
events_biosentinel.append(('2023-04-23T18:15:00', '2023-04-23T18:20', 26))
events_biosentinel.append(('2023-05-08T12:40:00', '2023-05-09T01:50', 38))
events_biosentinel.append(('2023-05-09T23:35:00', '2023-05-10T12:50', 83))
events_biosentinel.append(('2023-07-16T06:35:00', '2023-07-16T07:35', 18))
events_biosentinel.append(('2023-07-18T01:15:00', '2023-07-18T06:15', 620))
events_biosentinel.append(('2023-07-29T00:20:00', '2023-07-29T09:20', 154))
events_biosentinel.append(('2023-08-05T11:15:00', '2023-08-05T17:45', 18))
events_biosentinel.append(('2023-08-08T01:15:00', '2023-08-09T00:25', 47))
events_biosentinel.append(('2023-09-01T04:30:00', '2023-09-01T06:10', 25))
events_biosentinel.append(('2023-12-15T23:45:00', '2023-12-16T00:15', 13))
events_biosentinel.append(('2024-01-03T20:05:00', '2024-01-04T08:35', 20))
events_biosentinel.append(('2024-01-29T06:15:00', '2024-01-29T18:05', 137))
events_biosentinel.append(('2024-02-09T15:30:00', '2024-02-09T23:55', 187))
events_biosentinel.append(('2024-02-12T08:05:00', '2024-02-13T16:05', 118))
events_biosentinel.append(('2024-03-23T08:15:00', '2024-03-23T18:20', 956))
events_biosentinel.append(('2024-05-10T13:35:00', '2024-05-10T17:45', 208))
events_biosentinel.append(('2024-05-11T02:10:00', '2024-05-11T09:10', 116))

events_biosentinel = pd.DataFrame(events_biosentinel, columns=['begin', 'max', 'max_pfu'])

# Pre-begin and post-max durations to include, in multiples of the begin-max duration
pre_begin = 2
post_max = 6

events_biosentinel['duration'] = pd.to_datetime(events_biosentinel['max']) - pd.to_datetime(events_biosentinel['begin'])
events_biosentinel['date_start'] = pd.to_datetime(events_biosentinel['begin']) - pre_begin * events_biosentinel['duration']
events_biosentinel['date_end'] = pd.to_datetime(events_biosentinel['max']) + post_max * events_biosentinel['duration']

# Round up date_start to nearest :00, :15, :30, :45
events_biosentinel['date_start'] = events_biosentinel['date_start'].dt.ceil('15min')

def get_event_biosentinel(event_id):
    if event_id >= len(events_biosentinel):
        raise ValueError('Expecting 0 <= event_id < {}'.format(len(events_biosentinel)))
    event = events_biosentinel.iloc[event_id]
    date_start = event['date_start'].to_pydatetime().isoformat()
    date_end = event['date_end'].to_pydatetime().isoformat()
    max_pfu = event['max_pfu']
    return date_start, date_end, max_pfu

events = {}
for i in range(len(events_biosentinel)):
    # zero pad event_id to as many digits as there are events
    event_id = 'biosentinel' + str(i+1).zfill(len(str(len(events_biosentinel))))
    events[event_id] = get_event_biosentinel(i)

# event_id       date_start          date_end            max_pfu
# biosentinel01 2023-02-25T06:15:00 2023-02-28T01:40:00 58
# biosentinel02 2023-03-09T14:00:00 2023-03-26T09:25:00 22
# biosentinel03 2023-04-23T18:15:00 2023-04-23T18:50:00 26
# biosentinel04 2023-05-07T10:30:00 2023-05-12T08:50:00 38
# biosentinel05 2023-05-08T21:15:00 2023-05-13T20:20:00 83
# biosentinel06 2023-07-16T04:45:00 2023-07-16T13:35:00 18
# biosentinel07 2023-07-17T15:15:00 2023-07-19T12:15:00 620
# biosentinel08 2023-07-28T06:30:00 2023-07-31T15:20:00 154
# biosentinel09 2023-08-04T22:15:00 2023-08-07T08:45:00 18
# biosentinel10 2023-08-06T03:00:00 2023-08-14T19:25:00 47
# biosentinel11 2023-09-01T01:15:00 2023-09-01T16:10:00 25
# biosentinel12 2023-12-15T22:45:00 2023-12-16T03:15:00 13
# biosentinel13 2024-01-02T19:15:00 2024-01-07T11:35:00 20
# biosentinel14 2024-01-28T06:45:00 2024-02-01T17:05:00 137
# biosentinel15 2024-02-08T22:45:00 2024-02-12T02:25:00 187
# biosentinel16 2024-02-09T16:15:00 2024-02-21T16:05:00 118
# biosentinel17 2024-03-22T12:15:00 2024-03-26T06:50:00 956
# biosentinel18 2024-05-10T05:15:00 2024-05-11T18:45:00 208
# biosentinel19 2024-05-10T12:15:00 2024-05-13T03:10:00 116