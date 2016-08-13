%%%%%%%%%%%%%%%%%%%
%%% process and plot robot state from telemetry file, to be combined with
%%% shell data
%%%%%%%%%%%%%%%%%%%

%figure(3)
%clf

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
set(0,'DefaultTextFontSize', 14)

% time is in number of microseconds, sample rate is ~ 1ms, but some samples
% might be dropped in packets
time_sens = data(:,1);
rate = 1.0 * size(time_sens,1)/(time_sens(end)-time_sens(1));  % samples per microsecond
index = 1:size(data,1);

% moving average filter
filter_len = 10;
M_avg = zeros(size(data,1)-filter_len+1,size(data,2));
for col = 1:size(data,2)
    for row = 1:size(data,1)-filter_len+1
        M_avg(row,col) = mean(data(row:row+filter_len-1,col));
    end
end
index_avg = 1:size(data,1)-filter_len+1;
index_avg = index_avg+(filter_len-1)/2;
s = 4;
% maxt = ceil(index_avg(end)/rate)
maxt = time_sens(end)/1e6;

% ha = tight_subplot(5,1,[.02 0],[.1 .08],[.1 .01]);
% axes(ha(1)); plot(time(1:s:end)/1000,TorqueR(1:s:end),...
%     'r-','LineWidth',2);
% hold on; plot(time(1:s:end)/1000,-TorqueL(1:s:end),...
%     'b:','LineWidth',2);ylabel('\tau (mN-m)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-2.1,2.1]); legend('Right leg','Left leg')
% axes(ha(2)); plot(time(1:s:end)/1000,AngleZ(1:s:end),...
%     'k','LineWidth',3);ylabel('\theta_Z (radians)','FontSize', 18, 'FontName', 'CMU Serif');
%     axis([0,maxt,-.8,10]);% gyro Z angle
% axes(ha(3)); plot(time(1:s:end)/1000,AX(1:s:end),'k','LineWidth',2);
%     ylabel('x" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-15,15]);
% axes(ha(4)); plot(time(1:s:end)/1000,AY(1:s:end),'k','LineWidth',2);
%     ylabel('y" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([0,maxt,-15,15]);
% axes(ha(5)); plot(time(1:s:end)/1000,AZ(1:s:end),'k','LineWidth',2);
%     ylabel('z" (m/s^2)','FontSize', 18, 'FontName', 'CMU Serif');axis([-0,maxt,-5,25]);
% 
% set(ha(1:4),'XTickLabel','') % only 1 time lable
% for i = 1:4
%     axes(ha(i));
%     set(gca,'FontName','CMU Serif','FontSize',14);
%     %axis([0,10,-1.5,1.5]);
%     hold on
%     temp = get(gca,'XTick');
%     plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
%     grid OFF
%     
% end
% axes(ha(5));
% xlabel('Time (s)','FontSize', 18, 'FontName', 'CMU Serif');
% set(gcf,'Units','inches');
% set(gcf,'Position',[1 1 14 16]);
%pause(5);
%export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/force_and_leg_torque_and_theta' '-pdf' '-transparent' %-nocrop
%print(gcf,'-dpdf','/Users/jgoldberg/Desktop/test')

%%



