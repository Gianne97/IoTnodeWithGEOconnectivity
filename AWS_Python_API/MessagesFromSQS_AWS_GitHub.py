# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:36:34 2025

@author: giannetti
"""

import boto3
import base64
# from decimaldegrees import decimal2dms as dd
import decimaldegrees as dd
import numpy as np
import plotly.io as pio
import plotly.express as px
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
# import os
# import datetime
pio.renderers.default='browser'
# pio.renderers.default = "svg"

# %reset -f
# %clear

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
    # index_fPort = 20
    index_RawMessage = 21
    RawMessage = MessageBodySplit[index_RawMessage][16:len(MessageBodySplit[index_RawMessage])-1]
    # fPort = MessageBodySplit[index_fPort][len(MessageBodySplit[index_fPort])-2:len(MessageBodySplit[index_fPort])-1]

    # Message decoding
    base64_string = RawMessage
    base64_bytes = base64_string.encode("ascii")
    sample_string_bytes = base64.b64decode(base64_bytes)
    sample_string = sample_string_bytes.decode("ascii")

    # Sensor readings
    output = sample_string.split(",")
    
    idxMessID = 0
    MessID = output[idxMessID]
    
    # if fPort == '1':
    if len(output) == 6:
        
        idxGPSlat = 1
        idxGPSlon = 2
        idxNsat = 3
        idxAlt = 4
        idxHeight = 5
        
        if output[idxGPSlat] == '0':
            output[idxGPSlat] = output[idxGPSlon]
        elif output[idxGPSlon] == '0':
            output[idxGPSlon] = output[idxGPSlat]
        elif output[idxGPSlat] == '0' and output[idxGPSlon] == '0':
            output[idxGPSlat] = '400000000'
            output[idxGPSlon] = '100000000'
        
        GPSdm = [{ "degrees": int(output[idxGPSlat][0:2]), "minutes": float(output[idxGPSlat][2:4] + '.' + output[idxGPSlat][4:])}, { "degrees": int(output[idxGPSlon][0:2]), "minutes": float(output[idxGPSlon][2:4] + '.' + output[idxGPSlon][4:])}]
        Nsat = int(output[idxNsat])
        Alt = float(output[idxAlt])/10 # meters # https://www.fullyinstrumented.com/altitude-vs-elevation-vs-height/
        Height= float(output[idxHeight])/10 # meters
        TimeStamp = int(responseFilteredSpec['Attributes']['SentTimestamp'])
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(TimeStamp/1000))) # localtime # gmtime
        devEUI = MessageBodySplit[index_devEUI][11:len(MessageBodySplit[index_devEUI])-1]
        # Elevation Santa Marta: 96 m
        # https://www.advancedconverter.com/map-tools/find-altitude-by-coordinates
        # https://www.sunearthtools.com/dp/tools/conversion.php?lang=it
        # print(f"GPS data in dm format: {GPSdm}")
        GPSdecimal = [dd.dm2decimal(GPSdm[0]["degrees"], GPSdm[0]["minutes"]), dd.dm2decimal(GPSdm[1]["degrees"], GPSdm[1]["minutes"])]
        # print(f"GPS data in decimal format: {GPSdecimal}")
        # GPSdms = [dd.decimal2dms(GPSdecimal[0]), dd.decimal2dms(GPSdecimal[1])]
        # print(f"GPS data in dms format: {GPSdms}")
        
        Decoded = [GPSdecimal, Nsat, Alt, Height, formatted_time, devEUI, TimeStamp, MessID]
    else:
        idxSolRadIn = 1
        idxTemp = 2
        idxHumidity = 3
        idxAcc = 4
        
        # SolRadIn = int(output[idxSolRadIn]) # solar radiation (ADC from the control unit)
        # Vsupply = 3.3 # supply voltage from the control unit (V)
        # RangeADC = 2**12-1
        # VsolRad = Vsupply*SolRadIn/RangeADC # voltage sampled by the ADC (V)
        # RsolRad = 5.52e3 # resistance (ohm) # 5.593e3
        # IsolRad = (Vsupply - VsolRad)/RsolRad # current through the resistance (I)
        # ConvFactorSolRad = 2.6e4/0.5e-3 # conversion factor from the datasheet (lux/A)
        # SolRad = IsolRad * ConvFactorSolRad # final result (lux)
        
        SolRad = int(output[idxSolRadIn])
        
        Temp= int(output[idxTemp])/10 # Celsius degrees
        Humidity= int(output[idxHumidity])/10 # Percentage
        
        div = 2**14
        g = 9.80665
        Acc = np.array([float(output[idxAcc]), float(output[idxAcc+1]), float(output[idxAcc+2])])/div # Acceleration (m/s^2)
        AccMod = np.sqrt(sum(Acc**2))*g # offset +/-40 mg = 40*9.80665/1000 = 0.3923 m/s**2
        
        MessID = output[idxMessID]
        Decoded = [SolRad, Temp, Humidity, AccMod, MessID]
        
    return Decoded

#%% Start of the code

# %reset -f
# %clear

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

# outputFilename = 'MessagesFromSQS.json'
# file = open(outputFilename, 'r')
# responseFiltered = json.load(file)

#%%Processing

TS = []
fPort = []
for idx in range(len(responseFiltered)):
    responseFilteredSpec = responseFiltered[idx]
    # TS = int(responseFilteredSpec['Attributes']['SentTimestamp'])
    TS.append(int(responseFilteredSpec['Attributes']['SentTimestamp']))
    MessageBody = responseFilteredSpec['Body']
    MessageBodySplit = MessageBody.split(", ")
    index_devEUI = 3
    index_fPort = 20
    index_RawMessage = 21
    devEUI = MessageBodySplit[index_devEUI][11:len(MessageBodySplit[index_devEUI])-1]
    RawMessage = MessageBodySplit[index_RawMessage][16:len(MessageBodySplit[index_RawMessage])-1]
    fPort.append(MessageBodySplit[index_fPort][len(MessageBodySplit[index_fPort])-2:len(MessageBodySplit[index_fPort])-1])
    
Tot = [TS, fPort]
SortingIdx = np.argsort(TS)
fPortSorted = list(map(fPort.__getitem__, SortingIdx))
TSsorted = sorted(TS)
TotSorted = np.array([TSsorted, fPortSorted])

Decoded = []
MessIDdec = []
for idx in range(int(len(responseFiltered))):
    responseFilteredSpec = responseFiltered[idx]
    tmp = get_info(responseFilteredSpec)
    # if int(responseFilteredSpec['Attributes']['SentTimestamp']) == 1742382269064: # 1742382329177:
    #     print(idx)
    Decoded.append(tmp)
    MessIDdec.append(tmp[len(tmp)-1])

unique, SortingIdxMessIDdec, frequency = np.unique(MessIDdec, return_index=True, return_counts = True)
idx2Remove = []
for idx in range(len(frequency)):
    if frequency[idx] != 2:
        idx2Remove.append(idx)
        
NewFreq = np.delete(frequency, idx2Remove)
NewUnique = np.delete(unique, idx2Remove)

outputFilenameData = 'DataAll_v2.csv'

for idx in range(len(NewUnique)):
    idxMess = 0
    for ele in MessIDdec:
        if ele == NewUnique[idx]:
            if len(Decoded[idxMess]) == 8: # Decoded = [GPSdecimal, Nsat, Alt, Height, formatted_time, devEUI, TimeStamp, MessID]
                GPSdecimal = Decoded[idxMess][0]
                Nsat = Decoded[idxMess][1]
                Alt = Decoded[idxMess][2]
                Height = Decoded[idxMess][3]
                formatted_time = Decoded[idx][4]
                devEUI = Decoded[idxMess][5]
                TimeStamp = Decoded[idxMess][6]
                # if TimeStamp == 1742382329177:
                #     print(idxMess)
                
            else: # Decoded = [SolRad, Temp, Humidity, AccMod, MessID]
                
                SolRad = Decoded[idxMess][0]
                Temp = Decoded[idxMess][1]
                Humidity = Decoded[idxMess][2]
                AccMod = Decoded[idxMess][3]
                
        idxMess = idxMess + 1
    
    dataSingle = {'Lat': [float(GPSdecimal[0])],
                  'Long': [float(GPSdecimal[1])],
                  'Temperature': [Temp],
                  'Time': [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(TimeStamp/1000)))],
                  'Height': [Height],
                  'Altitude': [Alt],
                  'Humidity': [Humidity],
                  'Acceleration': [AccMod],
                  'TimeStamp': [TimeStamp],
                  'SolarRadiation': [SolRad],
                  'Nsat': [Nsat],
                  'devEUI': [devEUI]}
    
    df = pd.DataFrame(dataSingle)
    df.to_csv(outputFilenameData, mode='a', index=False, header=False)
    # if idx == 0:
    #     df.to_csv(outputFilenameData, index=False)
    # else:
    #     df.to_csv(outputFilenameData, mode='a', index=False, header=False)

#%% Section Map
# https://towardsdatascience.com/simple-gps-data-visualization-using-python-and-open-street-maps-50f992e9b676/?gi=8d9315a792f1

# df = pd.read_csv("Data.csv")

outputFilenameData = 'DataAll_v2.csv'
# outputFilenameData = 'DataAll_BeforeRemoval.csv'
df = pd.read_csv(outputFilenameData)

df.dropna(
    axis=0,
    how='any',
    # thresh=None,
    subset=None,
    inplace=True
)

TimeStampThreshold = 1742397919715 # 1742308363615 # 1742397919715 # 1742308363615 # 1742396400000 # 1742389200000 # 1742308363615 # 1742382000000 # 1742308363615 # 1742288326*1000 # 1742287155*1000 # 1740049066924 # 1739198642855
i = df[(df.TimeStamp < TimeStampThreshold)].index
df2plot = df.drop(i)

# for idx in range(len(df2plot)):
#     df2plot.LocalTime[idx] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(df2plot.TimeStamp[idx]/1000)))

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

#%% Plots

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12

xmin = 00
xmax = 00

ymin = 12
ymax = 20

# Plotting a line chart
ax = df2plot.plot(x='TimeStamp', y='Temperature', kind='scatter' , label="Temperature")
# ax.plot(df2plot.TimeStamp, df2plot.Humidity, 'b.')
ax.set_xlabel("Time stamp (ms)")
ax.set_ylabel("Temperature (Celsius degree)")
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
ax.annotate('Morning', xy=(1740051981059, 16), xytext=(1740051981059, 16))
ax.annotate('Afternoon', xy=(1740060154079, 18), xytext=(1740060154079, 18))
ax2 = ax.twinx()
ax2.plot(df2plot.TimeStamp, df2plot.Humidity, 'r.', label="Humidity")
# ax2.set_ylim([38,54])
ax2.set_ylabel("Humidity (%)")
# plt.title("Tests performed on February 20, 2025")
plt.grid()
ax.legend(["Temperature"], loc="lower left")
ax2.legend(["Humidity"], loc="lower right")
# plt.legend(["Temperature", "Humidity"], loc="lower right")
# f = plt.gcf()
# f.set_figwidth(6/2.54)
# f.set_figheight(4/2.54)
# plt.figure(figsize=(6/2.54, 6/2.54)) 
# plt.savefig("Tests20250220TemperatureHumidity.pdf", format='pdf')
plt.show()

#%%

# ax = df2plot.plot(x='TimeStamp', y='Temperature', secondary_y='Humidity', kind='scatter', label="Temperature")
# plt.show()

#%%

# # Plotting a line chart
# ax = df2plot.plot(x='TimeStamp', y='Humidity', kind='scatter')
# ax.set_xlabel("Time stamp (ms)")
# ax.set_ylabel("Humidity (%)")
# # ax.set_xlim([xmin, xmax])
# # ax.set_ylim([ymin, ymax])
# ax.annotate('Morning', xy=(1740051981059, 16), xytext=(1740051981059, 16))
# ax.annotate('Afternoon', xy=(1740062154079, 18), xytext=(1740062154079, 18))
# plt.title("Tests performed on February 20, 2025")
# plt.grid()
# # plt.savefig("Tests20250220Humidity.pdf", format='pdf')
# plt.show()

# Plotting a line chart
ax = df2plot.plot(x='TimeStamp', y='Acceleration', kind='scatter')
# fig, ax = plt.subplots(figsize=(1, 1))
# ax = plt.plot(df2plot.TimeStamp, df2plot.Acceleration)
ax.set_xlabel("Time stamp (ms)")
ax.set_ylabel("Acceleration (m/s^2)")
ax2 = ax.twinx()
ax2.plot(df2plot.TimeStamp, df2plot.SolarRadiation, 'r.', label="Humidity")
# ax2.set_ylim([38,54])
ax2.set_ylabel("Solar radiation (lux)")
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([5, 13])
ax.legend(["Acceleration"], loc="lower left")
ax2.legend(["Solar radiation"], loc="lower right")
ax.annotate('Morning', xy=(1740052995059, 10), xytext=(1740052995059, 10))
ax.annotate('Afternoon', xy=(1740062154079, 12), xytext=(1740062154079, 12))
# plt.title("Tests performed on February 20, 2025")
plt.grid()
# plt.savefig("Tests20250220AccSolar.pdf", format='pdf')
plt.show()