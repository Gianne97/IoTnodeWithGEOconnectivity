%% Polar plots

clear
close all

SimData_Tx = readmatrix("Data/NewABS_Screws/RealizedGain_Tx.csv");

RealizedGain2DAll_Phi0_Tx = readmatrix("Data/NewABS_Screws/RealizedGain2DAll_Phi0_Tx.csv");
RealizedGain2DAll_Phi90_Tx = readmatrix("Data/NewABS_Screws/RealizedGain2DAll_Phi90_Tx.csv");
RealizedGain2DAll_Theta90_Tx = readmatrix("Data/NewABS_Screws/RealizedGain2DAll_Theta90_Tx.csv");

MeasEIRP_Phi0 = load("Data\DATA_2025_04_24_SAGITTAL_PLANE_step3_proc.mat");
MeasEIRP_Phi90 = load("Data\DATA_2025_04_24_TRANSVERSAL_PLANE_step3_proc.mat");
MeasEIRP_Theta90 = load("Data\DATA_2025_04_24_EQUATORIAL_PLANE_step3_proc.mat");

maxTx = 10*log10(max(SimData_Tx(:,3)));

LimAll = 0.5;
Limits3D = [0, 1.2];

%% Tx tiled

Pin = 27; % Input power (dBm)
RlimitsEIRP = [-15, 30];

Rot_Theta90 = 90;
Rot_Phi90 = 0;
Rot_Phi0 = 0;

PhComp = 4;
ThComp = 5;
Total = 6;

Fontsize = 10;
Fontname = "TimesNewRoman";
TextAngle = deg2rad(220);
TextRadius = 60;

fig = figure;
t = tiledlayout(2,2,"TileSpacing","tight");

ax1 = nexttile(t);
ax1.Layout.Tile = 1;
% patternCustom(SimData_Tx(:,3),SimData_Tx(:,2),SimData_Tx(:,1));
% set(ax1,fontsize=12)
% clim(Limits3D);
% text(2.2,0.6,-0.4,"(a)","FontSize",Fontsize,"FontName",Fontname)

ax2 = polaraxes(t);
ax2.Layout.Tile = 2;
hold on
polarplot(deg2rad(RealizedGain2DAll_Theta90_Tx(:,3)), RealizedGain2DAll_Theta90_Tx(:,PhComp) + Pin)
polarplot(deg2rad(MeasEIRP_Theta90.angle_deg + Rot_Theta90), MeasEIRP_Theta90.Pt_Gt)
hold off
rlim(RlimitsEIRP)
ax2.RTick = -15:5:30;
ax2.RTickLabel = ["", "-10", "", "0", "", "10", "", "20", "", "30"];
title("\theta = 90°")
text(TextAngle,TextRadius,"(b)","FontSize",Fontsize,"FontName",Fontname)
max(RealizedGain2DAll_Theta90_Tx(:,PhComp) + Pin)
max(MeasEIRP_Theta90.Pt_Gt)

ax3 = polaraxes(t);
ax3.Layout.Tile = 3;
hold on
polarplot(deg2rad(RealizedGain2DAll_Phi90_Tx(:,3)), RealizedGain2DAll_Phi90_Tx(:,PhComp) + Pin)
polarplot(deg2rad(MeasEIRP_Phi90.angle_deg + Rot_Phi90), flip(MeasEIRP_Phi90.Pt_Gt))
hold off
rlim(RlimitsEIRP)
ax3.RTick = -15:5:30;
ax3.RTickLabel = ["", "-10", "", "0", "", "10", "", "20", "", "30"];
title("\phi = 90°")
text(TextAngle,TextRadius,"(c)","FontSize",Fontsize,"FontName",Fontname)
max(RealizedGain2DAll_Phi90_Tx(:,PhComp) + Pin)
max(MeasEIRP_Phi90.Pt_Gt)

ax4 = polaraxes(t);
ax4.Layout.Tile = 4;
hold on
polarplot(deg2rad(RealizedGain2DAll_Phi0_Tx(:,3)), RealizedGain2DAll_Phi0_Tx(:,ThComp) + Pin)
polarplot(deg2rad(MeasEIRP_Phi0.angle_deg + Rot_Phi0), MeasEIRP_Phi0.Pt_Gt)
hold off
rlim(RlimitsEIRP)
ax4.RTick = -15:5:30;
ax4.RTickLabel = ["", "-10", "", "0", "", "10", "", "20", "", "30"];
title("\phi = 0°")
text(TextAngle,TextRadius,"(d)","FontSize",Fontsize,"FontName",Fontname)
max(RealizedGain2DAll_Phi0_Tx(:,ThComp) + Pin)
max(MeasEIRP_Phi0.Pt_Gt)

set(ax1,"FontSize",Fontsize,"FontName",Fontname)
set(ax2,"FontSize",Fontsize,"FontName",Fontname)
set(ax3,"FontSize",Fontsize,"FontName",Fontname)
set(ax4,"FontSize",Fontsize,"FontName",Fontname)

% exportgraphics(fig,"results\EIRP_SimMeas.pdf")