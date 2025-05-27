# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:36:34 2025

@author: giannetti
"""

import boto3
import base64
import decimaldegrees as dd
import numpy as np
import plotly.io as pio
import plotly.express as px
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy
from scipy import stats

pio.renderers.default='browser'

#%% Definitions

def get_messages_from_queue(queue_url):
    sqs_client = boto3.client('sqs',region_name='eu-central-1',aws_access_key_id='YOUR_aws_access_key_id',aws_secret_access_key='YOUR_aws_secret_access_key')

    messages = []

    while True:
        resp = sqs_client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=['All'],
            MaxNumberOfMessages=10
        )

        try:
            messages.extend(resp['Messages'])
        except KeyError:
            break

        entries = [
            {'Id': msg['MessageId'], 'ReceiptHandle': msg['ReceiptHandle']}
            for msg in resp['Messages']
        ]

        resp = sqs_client.delete_message_batch(QueueUrl=queue_url, Entries=entries)

        if len(resp['Successful']) != len(entries):
            raise RuntimeError(
                f"Failed to delete messages: entries={entries!r} resp={resp!r}"
            )

    return messages

def get_info(responseFilteredSpec):
    MessageBody = responseFilteredSpec['Body']
    MessageBodySplit = MessageBody.split(", ")
    index_devEUI = 3
    index_RSSI = 5
    index_SNR = 6
    index_RawMessage = 21
    index_BW = 14
    index_frequency = 12
    freq = float(MessageBodySplit[index_frequency][14:len(MessageBodySplit[index_frequency])-1]) # Frequency (Hz)
    BW = float(MessageBodySplit[index_BW][14:len(MessageBodySplit[index_BW])-1]) # Bandwidth (kHz)
    RSSI = float(MessageBodySplit[index_RSSI][17:len(MessageBodySplit[index_RSSI])-1]) # received power (dBm)
    SNR = float(MessageBodySplit[index_SNR][16:len(MessageBodySplit[index_SNR])-1]) # (dB)
    RawMessage = MessageBodySplit[index_RawMessage][16:len(MessageBodySplit[index_RawMessage])-1]
        
    number_rec_hex = base64.b64decode(RawMessage).hex()

    idx_lat = [0,9]
    idx_long = [9,18]
    idx_sats = [18,20]
    idx_alt = [20,24]
    idx_heigh = [24,28]
    idx_solar = [28,32]
    idx_temp = [32,35]
    idx_hum = [35,38]
    idx_accx = [38,42]
    idx_accy = [42,46]
    idx_accz = [46,50]
    idx_time1 = [54,58] # TX
    idx_time2 = [50,54] # sleep
    idx_current1 = [64,70] # TX
    idx_current2 = [58,64] # sleep

    offset_lat = 900000000
    offset_long = 1800000000
    offset_sats = 0
    offset_alt = 10000
    offset_heigh = 10000
    offset_solar = 0
    offset_temp = 300
    offset_hum = 0
    offset_acc = 16384
    multiplier_time = 5 # mS
    multiplier_current = 0.010 # GPS (mA)
    multiplier_current = 0.10  # MODEM (mA)

    lat = int(number_rec_hex[idx_lat[0]:idx_lat[1]], 16) - offset_lat
    long = int(number_rec_hex[idx_long[0]:idx_long[1]], 16) - offset_long
    sats = int(number_rec_hex[idx_sats[0]:idx_sats[1]], 16) - offset_sats
    alt = int(number_rec_hex[idx_alt[0]:idx_alt[1]], 16) - offset_alt
    height = int(number_rec_hex[idx_heigh[0]:idx_heigh[1]], 16) - offset_heigh
    solar = int(number_rec_hex[idx_solar[0]:idx_solar[1]], 16) - offset_solar
    temp = int(number_rec_hex[idx_temp[0]:idx_temp[1]], 16) - offset_temp
    hum = int(number_rec_hex[idx_hum[0]:idx_hum[1]], 16) - offset_hum
    accx = int(number_rec_hex[idx_accx[0]:idx_accx[1]], 16) - offset_acc
    accy = int(number_rec_hex[idx_accy[0]:idx_accy[1]], 16) - offset_acc
    accz = int(number_rec_hex[idx_accz[0]:idx_accz[1]], 16) - offset_acc
    
    Nsample_TX = int(number_rec_hex[idx_time1[0]:idx_time1[1]], 16)
    cycle_time_TX = Nsample_TX*multiplier_time
    if Nsample_TX != 0:
        mean_current_TX = int(number_rec_hex[idx_current1[0]:idx_current1[1]], 16)*multiplier_current/Nsample_TX
        total_charge_TX = mean_current_TX*cycle_time_TX
    else:
        mean_current_TX = 0
        total_charge_TX = 0

    Nsample_sleep = int(number_rec_hex[idx_time2[0]:idx_time2[1]], 16)
    cycle_time_sleep = Nsample_sleep*multiplier_time
    if Nsample_sleep != 0:
        mean_current_sleep = int(number_rec_hex[idx_current2[0]:idx_current2[1]], 16)*multiplier_current/Nsample_sleep
        total_charge_sleep = mean_current_sleep*cycle_time_sleep
    else:
        mean_current_sleep = 0
        total_charge_sleep = 0
    
    lat_str = str(lat)
    long_str = str(long)
    
    GPSdm = [{ "degrees": int(lat_str[0:2]), "minutes": float(lat_str[2:4] + '.' + lat_str[4:])}, { "degrees": int(long_str[0:2]), "minutes": float(long_str[2:4] + '.' + long_str[4:])}]
    GPSdecimal = [dd.dm2decimal(GPSdm[0]["degrees"], GPSdm[0]["minutes"]), dd.dm2decimal(GPSdm[1]["degrees"], GPSdm[1]["minutes"])]
    Nsat = sats
    Alt = float(alt)/10
    Height = float(height)/10
    SolRad = solar
    Temp = float(temp)/10 # Celsius degrees
    Humidity = float(hum)/10
    
    div = offset_acc # 2**14
    g = 9.80665
    Acc = np.array([float(accx), float(accy), float(accz)])/div # Acceleration (m/s^2)
    AccMod = np.sqrt(sum(Acc**2))*g # offset +/-40 mg = 40*9.80665/1000 = 0.3923 m/s**2
    
    TimeStamp = int(responseFilteredSpec['Attributes']['SentTimestamp'])
    devEUI = MessageBodySplit[index_devEUI][11:len(MessageBodySplit[index_devEUI])-1]
    
    Decoded = [GPSdecimal, Nsat, Alt, Height, SolRad, Temp, Humidity, AccMod, devEUI, TimeStamp, freq, BW, RSSI, SNR, cycle_time_TX, mean_current_TX, total_charge_TX, cycle_time_sleep, mean_current_sleep, total_charge_sleep] # MessID must be the last entry
        
    return Decoded

#%% Start of the code

queue_url='YOUR_queue_url'

response = get_messages_from_queue(queue_url)

ThresholdMessage = 30
idxFiltered = 0
responseFiltered = []
for idx in range(len(response)):
    MessageBody = response[idx]['Body']
    MessageBodySplit = MessageBody.split(", ")
    index_RawMessage = 21
    if len(MessageBodySplit)>20:
        RawMessage = MessageBodySplit[index_RawMessage][16:len(MessageBodySplit[index_RawMessage])-1]
        # print(RawMessage)
        if len(RawMessage) > ThresholdMessage:
            responseFiltered.append(response[idx])
            idxFiltered = idxFiltered + 1

curr = str(time.time())
outputFilename = 'Data/SQS/MessagesFromSQS_' + curr + '.json'
file = open(outputFilename, 'w+')
json.dump(responseFiltered, file)

outputFilename = 'Data/SQS/MessagesFromSQS_NoFilter_' + curr + '.json'
file = open(outputFilename, 'w+')
json.dump(response, file)

#%%Processing

outputFilenameData = 'DataAll_v6.csv' # 2025/03/28

for idx in range(len(responseFiltered)):
    ProcData = get_info(responseFiltered[idx])
    
    # [GPSdecimal, Nsat, Alt, Height, SolRad, Temp, Humidity, AccMod, devEUI, TimeStamp, freq, BW, RSSI, SNR]    
    
    dataSingle = {'Lat': [float(ProcData[0][0])],
                  'Long': [float(ProcData[0][1])],
                  'Nsat': [ProcData[1]],
                  'Altitude': [ProcData[2]],
                  'Height': [ProcData[3]],
                  'SolarRadiation': [ProcData[4]],
                  'Temperature': [ProcData[5]],
                  'Humidity': [ProcData[6]],
                  'Acceleration': [ProcData[7]],
                  'devEUI': [ProcData[8]],
                  'TimeStamp': [ProcData[9]],
                  'Time': [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(ProcData[9]/1000)))],
                  'Frequency': [ProcData[10]],
                  'Bandwidth': [ProcData[11]],
                  'RSSI': [ProcData[12]],
                  'SNR': [ProcData[13]],
                  # 'CycleTime': [ProcData[14]],
                  # 'MeanCurrent': [ProcData[15]],
                  # 'TotalCharge': [ProcData[16]]
                  'CycleTimeTX': [ProcData[14]],
                  'MeanCurrentTX': [ProcData[15]],
                  'TotalChargeTX': [ProcData[16]],
                  'CycleTimeSleep': [ProcData[17]],
                  'MeanCurrentSleep': [ProcData[18]],
                  'TotalChargeSleep': [ProcData[19]]
                  }
        
    df = pd.DataFrame(dataSingle)
    df.to_csv(outputFilenameData, mode='a', index=False, header=False)
    # if idx == 0:
    #     df.to_csv(outputFilenameData, index=False)
    # else:
    #     df.to_csv(outputFilenameData, mode='a', index=False, header=False)

#%% Section Map

outputFilenameData = 'DataAll_v6.csv'
df = pd.read_csv(outputFilenameData)

df.dropna(
    axis=0,
    how='any',
    # thresh=None,
    subset=None,
    inplace=True
)

TimeStampThreshold = 1748329200000
i = df[(df.TimeStamp < TimeStampThreshold)].index
df2plot = df.drop(i)

TimeStampThresholdAbove = 2*1748329200000
i = df2plot[(df2plot.TimeStamp > TimeStampThresholdAbove)].index
df2plot = df2plot.drop(i)

# df2plot.to_csv('Data_GitHub_StandingTest.csv', index=False)
# df2plot.to_csv('Data_GitHub_MovingTest.csv', index=False)

# 1743401700000 - 1743444900000 # standing test
# 1743480900000 - 1743506100000 # moving test

# 1748257200000 - 1748275200000 # standing test 2025 05 26
# 1748241613833 - 1748250000000 # moving test 2025 05 26


DeviceID = '0016c001f0116140' # '0016c001f0116140'- Box B # '0016c001f01161bb' Box A
i = df2plot[(df2plot.devEUI != DeviceID)].index
df2plot = df2plot.drop(i)

NsatThreshold = 7 # 2*1742832849880
i = df2plot[(df2plot.Nsat <= NsatThreshold)].index
df2plot = df2plot.drop(i)

MinTimeStamp = df2plot['TimeStamp'].min()

Nmessages = df2plot.shape[0]

color_scale = [(0, 'orange'), (1,'red')]

fig = px.scatter_mapbox(df2plot, 
                        lat="Lat", 
                        lon="Long", 
                        hover_name="devEUI", 
                        hover_data=["devEUI", "Temperature", "Time", "Humidity", "Acceleration", "SolarRadiation", "TimeStamp"],
                        color="Temperature",
                        color_continuous_scale=color_scale,
                        size="Temperature",
                        zoom=8, 
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#%% Graphs all together

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10

TimeMin = (df2plot.TimeStamp-MinTimeStamp)/1000/60
TimeLabel = "Time (min)" # "Time stamp (ms)"

TempMin = df2plot['Temperature'].min()
TempMax = df2plot['Temperature'].max()

HumMin = df2plot['Humidity'].min()
HumMax = df2plot['Humidity'].max()

AccMean = df2plot['Acceleration'].mean()
AccStd  = df2plot['Acceleration'].std()

RSSImin = df2plot['RSSI'].min()
RSSImax = df2plot['RSSI'].max()

SNRmin = df2plot['SNR'].min()
SNRmax = df2plot['SNR'].max()

FreqUnique = df2plot['Frequency'].unique() - 2.009e9

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(df2plot['RSSI']), np.array(df2plot['SNR']))

fig, (ax1L, ax2L, ax3L) = plt.subplots(3,1)
fig.set_size_inches(8/2.54, 18/2.54)

yminLeft = 10
ymaxLeft = 30

yminRight = 35
ymaxRight = 55

ax1L.plot(TimeMin, df2plot.Temperature, 'b.', label="Temp.")
ax1L.set_ylabel("Temperature (°C)")
ax1R = ax1L.twinx()
ax1R.plot(TimeMin, df2plot.Humidity, 'r.', label="Rel. hum.")
ax1L.set_ylim([yminLeft, ymaxLeft])
ax1R.set_ylim([yminRight,ymaxRight])
ax1R.set_ylabel("Relative humidity (%)")
ax1L.grid(True)
lines, labels = ax1L.get_legend_handles_labels()
lines2, labels2 = ax1R.get_legend_handles_labels()
ax1R.legend(lines + lines2, labels + labels2, loc="lower left", ncol=2)

yminLeft = 4
ymaxLeft = 14

yminRight = 0
ymaxRight = 25

ax2L.plot(TimeMin, df2plot.Acceleration, 'b.', label="Acc.")
ax2L.set_ylabel("Acceleration ($\mathregular{m/s^{2}}$)")
ax2R = ax2L.twinx()
ax2R.plot(TimeMin, df2plot.SolarRadiation*1e-3, 'r.', label="Int. sol. rad.")
ax2L.set_ylim([yminLeft, ymaxLeft])
ax2R.set_ylim([yminRight,ymaxRight])
ax2R.set_ylabel("Intensity of solar radiation (klx)")
ax2L.grid(True)
lines, labels = ax2L.get_legend_handles_labels()
lines2, labels2 = ax2R.get_legend_handles_labels()
ax2R.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2)

yminLeft = -100
ymaxLeft = -60

yminRight = -20
ymaxRight = 20

ax3L.plot(TimeMin, df2plot.RSSI, 'b.', label="RSSI")
ax3L.set_ylabel("RSSI (dBm)")
ax3R = ax3L.twinx()
ax3R.plot(TimeMin, df2plot.SNR, 'r.', label="SNR")
ax3L.set_ylim([yminLeft, ymaxLeft])
ax3R.set_ylim([yminRight,ymaxRight])
ax3L.set_xlabel(TimeLabel)
ax3R.set_ylabel("SNR (dB)")
ax3L.grid(True)
lines, labels = ax3L.get_legend_handles_labels()
lines2, labels2 = ax3R.get_legend_handles_labels()
ax3R.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2)

# plt.savefig("Figures/StandingTest20250331.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/MovingTest20250331.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/MovingTest20250401.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/MovingTestSensors_20250526.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/StandingTestSensors_20250526.pdf", format='pdf', bbox_inches ="tight",)
plt.show()

#%% Current data - histograms

df2plot['mean_current_TX_corrected'] = df2plot.MeanCurrentTX - df2plot.MeanCurrentSleep
df2plot['total_charge_TX_corrected'] = df2plot.mean_current_TX_corrected*df2plot.CycleTimeTX

ZscoreThreshold = 1

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10

fig, (ax1L, ax2L, ax3L) = plt.subplots(3,1)
fig.set_size_inches(8/2.54, 27/2.54)

# plot a histogram
data_with_outliers = df2plot['CycleTimeTX']/1000
z_scores = stats.zscore(data_with_outliers)
data_without_outliers = data_with_outliers[(z_scores > -ZscoreThreshold) & (z_scores < ZscoreThreshold)]
MeanCycleTime = np.mean(data_without_outliers)

# Freedman-Diaconis rule
Q1 = np.percentile(data_without_outliers, 25)
Q3 = np.percentile(data_without_outliers, 75)
IQR = Q3 - Q1
BW = 2*IQR*len(data_without_outliers)**(-1/3)
Nbins = int(np.ceil((np.max(data_without_outliers) - np.min(data_without_outliers))/BW))

ax1L.hist(data_without_outliers, bins=Nbins)
ax1L.set_xlabel("Cycle time (s)")

data_with_outliers = df2plot['mean_current_TX_corrected']
z_scores = stats.zscore(data_with_outliers)
data_without_outliers = data_with_outliers[(z_scores > -ZscoreThreshold) & (z_scores < ZscoreThreshold)]
MeanCurrent = np.mean(data_without_outliers)

Q1 = np.percentile(data_without_outliers, 25)
Q3 = np.percentile(data_without_outliers, 75)
IQR = Q3 - Q1
BW = 2*IQR*len(data_without_outliers)**(-1/3)
Nbins = int(np.ceil((np.max(data_without_outliers) - np.min(data_without_outliers))/BW))

ax2L.hist(data_without_outliers, bins=Nbins)
ax2L.set_xlabel("Mean current (mA)")

data_with_outliers = df2plot['total_charge_TX_corrected']/1000
z_scores = stats.zscore(data_with_outliers)
data_without_outliers = data_with_outliers[(z_scores > -ZscoreThreshold) & (z_scores < ZscoreThreshold)]
MeanCharge = np.mean(data_without_outliers)

Q1 = np.percentile(data_without_outliers, 25)
Q3 = np.percentile(data_without_outliers, 75)
IQR = Q3 - Q1
BW = 2*IQR*len(data_without_outliers)**(-1/3)
Nbins = int(np.ceil((np.max(data_without_outliers) - np.min(data_without_outliers))/BW))

ax3L.hist(data_without_outliers, bins=Nbins)
ax3L.set_xlabel("Total charge (mC)")

# plt.savefig("Figures/MovingTestCurrentHist_20250526.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/StandingTestCurrentHist_20250526.pdf", format='pdf', bbox_inches ="tight",)
plt.show()

#%% Current data - time domain

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10

TimeMin = (df2plot.TimeStamp-MinTimeStamp)/1000/60
TimeLabel = "Time (min)" # "Time stamp (ms)"

fig, (ax1L, ax2L, ax3L) = plt.subplots(3,1)
fig.set_size_inches(8/2.54, 27/2.54)

yminLeft = 20
ymaxLeft = 70

ax1L.plot(TimeMin, df2plot.CycleTimeTX/1000, 'b.', label="CycleTimeTX")
ax1L.set_ylabel("Cycle time (s)")
ax1L.set_ylim([yminLeft, ymaxLeft])
ax1L.grid(True)
lines, labels = ax1L.get_legend_handles_labels()

yminLeft = 0
ymaxLeft = 60

ax2L.plot(TimeMin, df2plot.mean_current_TX_corrected, 'b.', label="Mean current (mA)")
ax2L.set_ylabel("Mean current (mA)")
ax2L.set_ylim([yminLeft, ymaxLeft])
ax2L.grid(True)
lines, labels = ax2L.get_legend_handles_labels()

yminLeft = 0
ymaxLeft = 2.5e3

ax3L.plot(TimeMin, df2plot.total_charge_TX_corrected/1000, 'b.', label="Total charge (mC)")
ax3L.set_ylabel("Total charge (mC)")
ax3L.set_ylim([yminLeft, ymaxLeft])
ax3L.set_xlabel(TimeLabel)
ax3L.grid(True)
lines, labels = ax3L.get_legend_handles_labels()

# plt.savefig("Figures/MovingTestCurrentCurves_20250526.pdf", format='pdf', bbox_inches ="tight",)
# plt.savefig("Figures/StandingTestCurrentCurves_20250526.pdf", format='pdf', bbox_inches ="tight",)
plt.show()