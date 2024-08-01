import pandas as pd

# BPD                   : 2022-11-16T11:00:00 - 2024-05-14T09:15:00
# sdoml-lite-biosentinel: 2022-11-01T00:00:00 - 2024-05-14T19:30:00
# CRaTER-D1D2           : 2009-06-26T21:00:00 - 2024-06-19T00:00:00

# Data taken from page: 
# Solar Proton Events Affecting the Earth Environment
# https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html

# Begin Time Yr M/D (UTC); Maximum Time Yr M/D (UTC); >10 MeV Maximum (pfu)
events = []
events.append(('biosentinel', '2023-02-25T21:10:00', '2023-02-26T04:40', 58)) # Also in crater
events.append(('biosentinel', '2023-03-13T07:35:00', '2023-03-15T04:25', 22)) # Also in crater
events.append(('biosentinel', '2023-04-23T18:15:00', '2023-04-23T18:20', 26)) # Also in crater
events.append(('biosentinel', '2023-05-08T12:40:00', '2023-05-09T01:50', 38)) # Also in crater
events.append(('biosentinel', '2023-05-09T23:35:00', '2023-05-10T12:50', 83)) # Also in crater
events.append(('biosentinel', '2023-07-16T06:35:00', '2023-07-16T07:35', 18)) # Also in crater
events.append(('biosentinel', '2023-07-18T01:15:00', '2023-07-18T06:15', 620)) # Also in crater
events.append(('biosentinel', '2023-07-29T00:20:00', '2023-07-29T09:20', 154)) # Also in crater
events.append(('biosentinel', '2023-08-05T11:15:00', '2023-08-05T17:45', 18)) # Also in crater
events.append(('biosentinel', '2023-08-08T01:15:00', '2023-08-09T00:25', 47)) # Also in crater
events.append(('biosentinel', '2023-09-01T04:30:00', '2023-09-01T06:10', 25)) # Also in crater
events.append(('biosentinel', '2023-12-15T23:45:00', '2023-12-16T00:15', 13)) # Also in crater
events.append(('biosentinel', '2024-01-03T20:05:00', '2024-01-04T08:35', 20)) # Also in crater
events.append(('biosentinel', '2024-01-29T06:15:00', '2024-01-29T18:05', 137)) # Also in crater
events.append(('biosentinel', '2024-02-09T15:30:00', '2024-02-09T23:55', 187)) # Also in crater
events.append(('biosentinel', '2024-02-12T08:05:00', '2024-02-13T16:05', 118)) # Also in crater
events.append(('biosentinel', '2024-03-23T08:15:00', '2024-03-23T18:20', 956)) # Also in crater
events.append(('biosentinel', '2024-05-10T13:35:00', '2024-05-10T17:45', 208)) # Also in crater
events.append(('biosentinel', '2024-05-11T02:10:00', '2024-05-11T09:10', 116)) # Also in crater

events.append(('crater', '2010-08-14T12:30:00', '2010-08-14T12:45:00', 14))
events.append(('crater', '2011-03-08T01:05:00', '2011-03-08T08:00:00', 50))
events.append(('crater', '2011-03-21T19:50:00', '2011-03-22T01:35:00', 14))
events.append(('crater', '2011-06-07T08:20:00', '2011-06-07T18:20:00', 73))
events.append(('crater', '2011-08-04T06:35:00', '2011-08-05T21:50:00', 96))
events.append(('crater', '2011-08-09T08:45:00', '2011-08-09T12:10:00', 26))
events.append(('crater', '2011-09-23T22:55:00', '2011-09-26T11:55:00', 35))
events.append(('crater', '2011-11-26T11:25:00', '2011-11-27T01:25:00', 80))
events.append(('crater', '2012-01-23T05:30:00', '2012-01-24T15:30:00', 6310))
events.append(('crater', '2012-01-27T19:05:00', '2012-01-28T02:05:00', 796))
events.append(('crater', '2012-03-07T05:10:00', '2012-03-08T11:15:00', 6530))
events.append(('crater', '2012-03-13T18:10:00', '2012-03-13T20:45:00', 469))
events.append(('crater', '2012-05-17T02:10:00', '2012-05-17T04:30:00', 255))
events.append(('crater', '2012-05-27T05:35:00', '2012-05-27T10:45:00', 14))
events.append(('crater', '2012-06-16T19:55:00', '2012-06-16T20:20:00', 14))
events.append(('crater', '2012-07-07T04:00:00', '2012-07-07T07:45:00', 25))
events.append(('crater', '2012-07-09T01:30:00', '2012-07-09T04:30:00', 19))
events.append(('crater', '2012-07-12T18:35:00', '2012-07-12T22:25:00', 96))
events.append(('crater', '2012-07-17T17:15:00', '2012-07-18T06:00:00', 136))
events.append(('crater', '2012-07-23T15:45:00', '2012-07-23T21:45:00', 12))
events.append(('crater', '2012-09-01T13:35:00', '2012-09-02T08:50:00', 60))
events.append(('crater', '2012-09-28T03:00:00', '2012-09-28T04:45:00', 28))
events.append(('crater', '2013-03-16T19:40:00', '2013-03-17T07:00:00', 16))
events.append(('crater', '2013-04-11T10:55:00', '2013-04-11T16:45:00', 114))
events.append(('crater', '2013-05-15T13:25:00', '2013-05-17T17:20:00', 42))
events.append(('crater', '2013-05-22T14:20:00', '2013-05-23T06:50:00', 1660))
events.append(('crater', '2013-06-23T20:14:00', '2013-06-24T05:20:00', 14))
events.append(('crater', '2013-09-30T05:05:00', '2013-09-30T20:05:00', 182))
events.append(('crater', '2013-12-28T21:50:00', '2013-12-28T23:15:00', 29))
events.append(('crater', '2014-01-06T09:15:00', '2014-01-06T16:00:00', 42))
events.append(('crater', '2014-01-06T09:15:00', '2014-01-09T03:40:00', 1026))
events.append(('crater', '2014-02-20T08:50:00', '2014-02-20T09:25:00', 22))
events.append(('crater', '2014-02-25T13:55:00', '2014-02-25T08:45:00', 103))
events.append(('crater', '2014-04-18T15:25:00', '2014-04-19T01:05:00', 58))
events.append(('crater', '2014-09-11T02:40:00', '2014-09-12T15:55:00', 126))
events.append(('crater', '2015-06-18T11:35:00', '2015-06-18T14:45:00', 17))
events.append(('crater', '2015-06-21T21:35:00', '2015-06-22T19:00:00', 1066))
events.append(('crater', '2015-06-26T03:50:00', '2015-06-27T00:30:00', 22))
events.append(('crater', '2015-10-29T05:50:00', '2015-10-29T10:00:00', 23))
events.append(('crater', '2016-01-02T04:30:00', '2016-01-02T04:50:00', 21))
events.append(('crater', '2017-07-14T09:00:00', '2017-07-14T23:20:00', 22))
events.append(('crater', '2017-09-05T00:40:00', '2017-09-08T00:35:00', 844))
events.append(('crater', '2017-09-10T16:45:00', '2017-09-11T11:45:00', 1494))
events.append(('crater', '2021-05-29T03:00:00', '2021-05-29T03:20:00', 15))
events.append(('crater', '2021-10-28T16:35:00', '2021-10-29T02:50:00', 29))
events.append(('crater', '2021-10-30T21:00:00', '2021-10-30T21:05:00', 11))
events.append(('crater', '2022-03-28T13:25:00', '2022-03-28T14:50:00', 19))
events.append(('crater', '2022-03-31T06:20:00', '2022-03-31T06:30:00', 11))
events.append(('crater', '2022-04-02T14:30:00', '2022-04-02T16:00:00', 32))
events.append(('crater', '2022-08-27T11:55:00', '2022-08-27T12:20:00', 27))
events.append(('crater', '2023-02-25T21:10:00', '2023-02-26T04:40', 58)) # Also in biosentinel
events.append(('crater', '2023-03-13T07:35:00', '2023-03-15T04:25', 22)) # Also in biosentinel
events.append(('crater', '2023-04-23T18:15:00', '2023-04-23T18:20', 26)) # Also in biosentinel
events.append(('crater', '2023-05-08T12:40:00', '2023-05-09T01:50', 38)) # Also in biosentinel
events.append(('crater', '2023-05-09T23:35:00', '2023-05-10T12:50', 83)) # Also in biosentinel
events.append(('crater', '2023-07-16T06:35:00', '2023-07-16T07:35', 18)) # Also in biosentinel
events.append(('crater', '2023-07-18T01:15:00', '2023-07-18T06:15', 620)) # Also in biosentinel
events.append(('crater', '2023-07-29T00:20:00', '2023-07-29T09:20', 154)) # Also in biosentinel
events.append(('crater', '2023-08-05T11:15:00', '2023-08-05T17:45', 18)) # Also in biosentinel
events.append(('crater', '2023-08-08T01:15:00', '2023-08-09T00:25', 47)) # Also in biosentinel
events.append(('crater', '2023-09-01T04:30:00', '2023-09-01T06:10', 25)) # Also in biosentinel
events.append(('crater', '2023-12-15T23:45:00', '2023-12-16T00:15', 13)) # Also in biosentinel
events.append(('crater', '2024-01-03T20:05:00', '2024-01-04T08:35', 20)) # Also in biosentinel
events.append(('crater', '2024-01-29T06:15:00', '2024-01-29T18:05', 137)) # Also in biosentinel
events.append(('crater', '2024-02-09T15:30:00', '2024-02-09T23:55', 187)) # Also in biosentinel
events.append(('crater', '2024-02-12T08:05:00', '2024-02-13T16:05', 118)) # Also in biosentinel
events.append(('crater', '2024-03-23T08:15:00', '2024-03-23T18:20', 956)) # Also in biosentinel
events.append(('crater', '2024-05-10T13:35:00', '2024-05-10T17:45', 208)) # Also in biosentinel
events.append(('crater', '2024-05-11T02:10:00', '2024-05-11T09:10', 116)) # Also in biosentinel

events = pd.DataFrame(events, columns=['prefix', 'begin', 'max', 'max_pfu'])

# sort by begin
events = events.sort_values(by='begin')

# Pre-begin and post-max durations to include, in multiples of the begin-max duration
pre_begin = 2
post_max = 6

format = 'ISO8601'
events['duration'] = pd.to_datetime(events['max'], format=format) - pd.to_datetime(events['begin'], format=format)
events['date_start'] = pd.to_datetime(events['begin'], format=format) - pre_begin * events['duration']
events['date_end'] = pd.to_datetime(events['max'], format=format) + post_max * events['duration']

# Round up date_start to nearest :00, :15, :30, :45
events['date_start'] = events['date_start'].dt.ceil('15min')

# count the number of times each unique prefix appears
events['event_id'] = events.groupby('prefix').cumcount() + 1

events_dict = {}
for prefix in events['prefix'].unique():
    events_with_prefix = events[events['prefix'] == prefix]
    num_events = len(events_with_prefix)
    for i in range(num_events):
        event = events_with_prefix.iloc[i]
        event_id = prefix + str(i+1).zfill(len(str(num_events)))
        date_start = event['date_start'].isoformat()
        date_end = event['date_end'].isoformat()
        max_pfu = event['max_pfu']
        events_dict[event_id] = date_start, date_end, max_pfu

events = events_dict

# for event, val in events.items():
#     print(event, val[0], val[1], val[2])

# event_id      date_start          date_end            max_pfu
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
# crater01      2010-08-14T12:00:00 2010-08-14T14:15:00 14
# crater02      2011-03-07T11:15:00 2011-03-10T01:30:00 50
# crater03      2011-03-21T08:30:00 2011-03-23T12:05:00 14
# crater04      2011-06-06T12:30:00 2011-06-10T06:20:00 73
# crater05      2011-08-01T00:15:00 2011-08-15T17:20:00 96
# crater06      2011-08-09T02:00:00 2011-08-10T08:40:00 26
# crater07      2011-09-18T21:00:00 2011-10-11T17:55:00 35
# crater08      2011-11-25T07:30:00 2011-11-30T13:25:00 80
# crater09      2012-01-20T09:30:00 2012-02-02T03:30:00 6310
# crater10      2012-01-27T05:15:00 2012-01-29T20:05:00 796
# crater11      2012-03-04T17:00:00 2012-03-15T23:45:00 6530
# crater12      2012-03-13T13:00:00 2012-03-14T12:15:00 469
# crater13      2012-05-16T21:30:00 2012-05-17T18:30:00 255
# crater14      2012-05-26T19:15:00 2012-05-28T17:45:00 14
# crater15      2012-06-16T19:15:00 2012-06-16T22:50:00 14
# crater16      2012-07-06T20:30:00 2012-07-08T06:15:00 25
# crater17      2012-07-08T19:30:00 2012-07-09T22:30:00 19
# crater18      2012-07-12T11:00:00 2012-07-13T21:25:00 96
# crater19      2012-07-16T15:45:00 2012-07-21T10:30:00 136
# crater20      2012-07-23T03:45:00 2012-07-25T09:45:00 12
# crater21      2012-08-30T23:15:00 2012-09-07T04:20:00 60
# crater22      2012-09-27T23:30:00 2012-09-28T15:15:00 28
# crater23      2013-03-15T21:00:00 2013-03-20T03:00:00 16
# crater24      2013-04-10T23:15:00 2013-04-13T03:45:00 114
# crater25      2013-05-11T05:45:00 2013-05-30T16:50:00 42
# crater26      2013-05-21T05:30:00 2013-05-27T09:50:00 1660
# crater27      2013-06-23T02:15:00 2013-06-26T11:56:00 14
# crater28      2013-09-28T23:15:00 2013-10-04T14:05:00 182
# crater29      2013-12-28T19:00:00 2013-12-29T07:45:00 29
# crater30      2014-01-05T19:45:00 2014-01-08T08:30:00 42
# crater31      2013-12-31T20:30:00 2014-01-25T18:10:00 1026
# crater32      2014-02-20T07:45:00 2014-02-20T12:55:00 22
# crater33      2014-02-26T00:15:00 2014-02-24T01:45:00 103
# crater34      2014-04-17T20:15:00 2014-04-21T11:05:00 58
# crater35      2014-09-08T00:15:00 2014-09-21T23:25:00 126
# crater36      2015-06-18T05:15:00 2015-06-19T09:45:00 17
# crater37      2015-06-20T02:45:00 2015-06-28T03:30:00 1066
# crater38      2015-06-24T10:30:00 2015-07-02T04:30:00 22
# crater39      2015-10-28T21:30:00 2015-10-30T11:00:00 23
# crater40      2016-01-02T04:00:00 2016-01-02T06:50:00 21
# crater41      2017-07-13T04:30:00 2017-07-18T13:20:00 22
# crater42      2017-08-30T01:00:00 2017-09-26T00:05:00 844
# crater43      2017-09-09T02:45:00 2017-09-16T05:45:00 1494
# crater44      2021-05-29T02:30:00 2021-05-29T05:20:00 15
# crater45      2021-10-27T20:15:00 2021-10-31T16:20:00 29
# crater46      2021-10-30T21:00:00 2021-10-30T21:35:00 11
# crater47      2022-03-28T10:45:00 2022-03-28T23:20:00 19
# crater48      2022-03-31T06:00:00 2022-03-31T07:30:00 11
# crater49      2022-04-02T11:30:00 2022-04-03T01:00:00 32
# crater50      2022-08-27T11:15:00 2022-08-27T14:50:00 27
# crater51      2023-02-25T06:15:00 2023-02-28T01:40:00 58
# crater52      2023-03-09T14:00:00 2023-03-26T09:25:00 22
# crater53      2023-04-23T18:15:00 2023-04-23T18:50:00 26
# crater54      2023-05-07T10:30:00 2023-05-12T08:50:00 38
# crater55      2023-05-08T21:15:00 2023-05-13T20:20:00 83
# crater56      2023-07-16T04:45:00 2023-07-16T13:35:00 18
# crater57      2023-07-17T15:15:00 2023-07-19T12:15:00 620
# crater58      2023-07-28T06:30:00 2023-07-31T15:20:00 154
# crater59      2023-08-04T22:15:00 2023-08-07T08:45:00 18
# crater60      2023-08-06T03:00:00 2023-08-14T19:25:00 47
# crater61      2023-09-01T01:15:00 2023-09-01T16:10:00 25
# crater62      2023-12-15T22:45:00 2023-12-16T03:15:00 13
# crater63      2024-01-02T19:15:00 2024-01-07T11:35:00 20
# crater64      2024-01-28T06:45:00 2024-02-01T17:05:00 137
# crater65      2024-02-08T22:45:00 2024-02-12T02:25:00 187
# crater66      2024-02-09T16:15:00 2024-02-21T16:05:00 118
# crater67      2024-03-22T12:15:00 2024-03-26T06:50:00 956
# crater68      2024-05-10T05:15:00 2024-05-11T18:45:00 208
# crater69      2024-05-10T12:15:00 2024-05-13T03:10:00 116