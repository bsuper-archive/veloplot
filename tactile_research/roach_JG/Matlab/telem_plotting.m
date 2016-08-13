%%%%%%%%%%%%%%%%%%%
%%% early version - does not read raw telemetry file - needs processed file
%%%%%%%%%%%%%%%%%%%

disp('!!requires telem file in .txt format!!')

clear % clear all variables between plots
figure(1)
clf
%data_file ='./2-28-15_data/behavior4_2Hz.txt'
%data_file ='../python/Data/2015.12.30_16.09.44_trial_imudata.txt';
%data_file ='../python/Data/2015.12.26_19.00.08_trial_imudata.txt';
%data = csvread('./2-28-15_data/behavior4_2Hz.txt',9,0);
data = csvread('../Python/Data/2016.03.03_17.44.35_nomove_inverted.txt')
legScale = 95.8738e-6; % 16 bit to radian
vref = 3.3; % for voltage conversion
vdivide = 3.7/2.7;  % for battery scaling
vgain = 15.0/47.0;  % gain of differential amplifier
RMotor = 3.3;   % resistance for SS7-3.3 ** need to check **
Kt = 1.41; %  motor torque constant mN-m/A  ** SS7-3.3 **

%acelerometer scale in mpu6000.c set to +- 8g
% +- 32768 data
xlScale = (1/4096.0) * 9.81;

% gyro in mpu6000.c scale set to +-2000 degrees per second
% +- 32768
gyroScale = (1/16.384) * (pi/180.0);  

length = 16;
width = 8;

% Timestamps
time = data(:,1)/1000.0;

% Leg Position R,L
rightLegPos = data(:,2)*legScale;
leftLegPos = data(:,3)*legScale;

% Commanded Leg Position R,L
commandedRightLegPos = data(:,4)*legScale;
commandedLeftLegPos = data(:,5)*legScale;

% Duty Cycle R,L
%DCR = -data[:,5]/4000.0
%DCL = -data[:,6]/4000.0
%DCR[DCR < -4000.0] = -4000.0
%DCL[DCL < -4000.0] = -4000.0
% raw 12 bit PWM value. Max = 3900.
DCR = data(:,6);
DCL = data(:,7);

% Gyro X,Y,Z
GyroX = data(:,8)*gyroScale;
GyroY = data(:,9)*gyroScale;
GyroZ = data(:,10)*gyroScale;

% Accelerometer X,Y,Z
AX = data(:,11)*xlScale;
AY = data(:,12)*xlScale;
AZ = data(:,13)*xlScale;

% BackEMF R,L
% A/D data is 10 bits, Vref+ = AVdd = 3.3V, VRef- = AVss = 0.0V
% BEMF volts = (15K)/(47K) * Vm + vref/2 - pidObjs[i].inputOffset
%RBEMF = -data[:,13]*vdivide*vref/1023.0
%LBEMF = -data[:,14]*vdivide*vref/1023.0
RBEMF = data(:,14)*vref/1024.0/vgain;  %# scale A/D to 0 to 3.3 V range and undo diff amp gain
LBEMF = data(:,15)*vref/1024/vgain;

% Battery Voltage in volts
VBatt = data(:,16)*vdivide*vref/1023.0;

%Power calculation
% i_m = (VBatt - BEMF)/R
% V_m is just VBatt
PowerR = abs((DCR/4096.0).*VBatt.*(VBatt - RBEMF)/RMotor); %# P = V_m i_m x duty cycle
PowerL = abs((DCL/4096.0).*VBatt.*(VBatt - LBEMF)/RMotor); %# P = V_m i_m x duty cycle

Energy = zeros(size(VBatt,1),1);
dt = (time(2) - time(1)) / 1000.0; %# time in seconds
%print 'dt=', dt
%energy calculation
for i = 2:size(VBatt,1)
    Energy(i) = Energy(i-1) + (PowerR(i) + PowerL(i)) * dt;
end

% torque calculation
TorqueR = (DCR/4096.0).*Kt.*(VBatt - RBEMF)/RMotor; % # \Tau = Kt i_m x duty cycle
TorqueL = (DCL/4096.0).*Kt.*(VBatt - LBEMF)/RMotor; % # \Tau = Kt i_m x duty cycle
    
% Motor Voltage, R,L
VMotorR = VBatt.*DCR;
VMotorL = VBatt.*DCL;

%added
%approximate integral of angle
AngleZ = zeros(size(GyroZ,1),1);
for i = 2:size(GyroZ,1)
   AngleZ(i) = AngleZ(i-1) + GyroZ(i)/1000.0;
end



%%
% Change default text fonts.
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultTextFontSize', 18)

%M = csvread('/Users/jgoldberg/Dropbox/Research/2-28-15_data/behavior4_2Hz.csv');
M = csvread('./2-28-15_data/behavior4_2Hz.csv');
%M = csvread(data_file,9,0);  % should be procesed csv file?
load('N_matrix_trial5.mat');
time_sens = M(:,1);
rate = 1.0 * size(time_sens,1)/(time_sens(end)-time_sens(1));  % samples per microsecond
index = 1:size(M,1);

% moving average filter
filter_len = 10;
M_avg = zeros(size(M,1)-filter_len+1,size(M,2));
for col = 1:size(M,2)
    for row = 1:size(M,1)-filter_len+1
        M_avg(row,col) = mean(M(row:row+filter_len-1,col));
    end
end
index_avg = 1:size(M,1)-filter_len+1;
index_avg = index_avg+(filter_len-1)/2;
%M_avg = M; %for no filtering
%index_avg = index;

% calculate forces
S = M_avg(:,2:9);
A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
F = A * N;

s = 4;
% maxt = ceil(index_avg(end)/rate)
maxt = 10;
ha = tight_subplot(5,1,[.02 0],[.1 .08],[.1 .01]);
axes(ha(1)); plot(index_avg./rate,F(:,1)-mean(F(50:100,1)),...
    'k','LineWidth',3);ylabel('Fx (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-0.8,.2]);
%title('Force and leg torque','FontSize', 24, 'FontName', 'CMU Serif');
axes(ha(2)); plot(index_avg./rate,F(:,2)-mean(F(50:100,2)),...
    'k','LineWidth',3);ylabel('Fy (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-.3,.5]);
axes(ha(3)); plot(index_avg./rate,F(:,3)-mean(F(50:100,3)),...
    'k','LineWidth',3);ylabel('Fz (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-.7,.3]);
axes(ha(4)); plot(time(1:s:end)/1000,TorqueR(1:s:end),...
    'r-','LineWidth',3);%ylabel('T_R (mN-m)','FontSize', 18, 'FontName', 'CMU Serif');%axis([0,10,-2,4]);
hold on; plot(time(1:s:end)/1000,-TorqueL(1:s:end),...
    'b:','LineWidth',3);ylabel('\tau (mN-m)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-2.1,2.1]); legend('Right leg','Left leg')
axes(ha(5)); plot(time(1:s:end)/1000,AngleZ(1:s:end),...
    'k','LineWidth',3);ylabel('\theta_Z (radians)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-.8,.3]);% legend('Right leg','Left leg')

set(ha(1:4),'XTickLabel',''); %set(ha,'YTickLabel','')
for i = 1:5
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',18);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid OFF
    
end

xlabel('Time (s)','FontSize', 18, 'FontName', 'CMU Serif');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
%pause(5);
%export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/force_and_leg_torque_and_theta' '-pdf' '-transparent' %-nocrop
%print(gcf,'-dpdf','/Users/jgoldberg/Desktop/test')

%%
% plot force data vs accelerometer- good for gravity static calibration
figure(2)
clf

% Change default text fonts.
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultTextFontSize', 18)

load('N_matrix_trial5.mat');
time_sens = M(:,1);
rate = size(time_sens,1)/(time_sens(end)-time_sens(1));
index = 1:size(M,1);

% moving average filter
filter_len = 10;
M_avg = zeros(size(M,1)-filter_len+1,size(M,2));
for col = 1:size(M,2)
    for row = 1:size(M,1)-filter_len+1
        M_avg(row,col) = mean(M(row:row+filter_len-1,col));
    end
end
index_avg = 1:size(M,1)-filter_len+1;
index_avg = index_avg+(filter_len-1)/2;
%M_avg = M; %for no filtering
%index_avg = index;

% calculate forces
S = M_avg(:,2:9);
A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
F = A * N;

s = 4;

ha = tight_subplot(6,1,[.04 0],[.1 .08],[.1 .01]);
axes(ha(1)); plot(index_avg./rate,F(:,1)-mean(F(50:100,1)),'k','LineWidth',3);ylabel('Fx (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,-0.8,.2]);
%title('Force and leg torque','FontSize', 24, 'FontName', 'CMU Serif');
axes(ha(2)); plot(index_avg./rate,F(:,2)-mean(F(50:100,2)),'k','LineWidth',3);ylabel('Fy (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,-.5,.5]);
axes(ha(3)); plot(index_avg./rate,F(:,3)-mean(F(50:100,3)),'k','LineWidth',3);ylabel('Fz (N)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,-.7,.3]);
axes(ha(4)); plot(time(1:s:end)/1000,AX(1:s:end),'k','LineWidth',3);ylabel('x" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,-10,10]);
axes(ha(5)); plot(time(1:s:end)/1000,AY(1:s:end),'k','LineWidth',3);ylabel('y" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,-10,10]);
axes(ha(6)); plot(time(1:s:end)/1000,AZ(1:s:end),'k','LineWidth',3);ylabel('z" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,10,0,20]);

set(ha(1:5),'XTickLabel',''); %set(ha,'YTickLabel','')
for i = 1:6
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',18);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid OFF
        
end
%r1 = rectangle('Position',[0,-5,2,10],'FaceColor','k');

xlabel('Time (s)','FontSize', 18, 'FontName', 'CMU Serif');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
%pause(3);
%export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/force_and_leg_torque' '-pdf' '-transparent' %-nocrop
%print(gcf,'-dpdf','/Users/jgoldberg/Desktop/test')



