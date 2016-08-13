%close all
%clear all

%%%%%%%%%%%%%%%
% plot including synchronize with leg cycle

%load('D:\Dropbox\Work\Grass\working\robot\shell force\5-11-15_data_with_chen\N_matrix_trial9.mat')
load('../telemetry/N_matrix_trial9.mat')
% T = csvread('velociroach_s=3cm_w=3cm_h=10cm_layer=5_f=10Hz_beetleshell_run3.txt',9,0);
%T = csvread('../telemetry/alldata/velociroach_s=10cm_w=5.5cm_h=27cm_layer=3_ply=6_f=13Hz_beetleshell_run3.txt',9,0);
%T = csvread('../../roach_JG/python/Data/2016.03.03_17.44.35_nomove_inverted.txt',9,0);
% T = csvread('../../roach_JG/python/Data/2016.03.03_19.11.13_tile0.4Hz_35grambox.txt',9,0)
%filename = '../../roach_JG/python/Data/2016.03.12_17.10.37_trial_imudata.txt'
filename = '../../../../GroupMeet/SkinProc/2016.03.06_19.44.56_trial_3_imudata.txt'
%filename = '../../../../GroupMeet/SkinProc/2016.03.06_19.16.20_trial_3_imudata.txt'
%filename = '../../../../GroupMeet/SkinProc/sliding5.txt'
T = csvread(filename,9,0);
data = T;  % data is used by state_plot
state_plot  % process robot state information from telemetry file
VREP = 1 %  use telem data from vrep which has force torque data already in column form
if (VREP == 1)
    S = T(:,17:24);
    A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
    Frecov = A*N;
    Frecov1 = Frecov;
    i = 2;
else
    Frecov1=T(:,17:22);
end
%display('line 21')
%return; % debugging
%eliminate duplicates
while 0
    Flen = size(Frecov1,1);
    if Flen < i
        break
    end
    if sum(Frecov1(i,:) == Frecov1(i-1,:)) == 6
        Frecov1 = [Frecov1(1:i-1,:);Frecov1(i+1:end,:)];
    else
        i = i + 1;
    end
end

T=T/1000000;
Frecov1offset=mean(Frecov1(1:50,:));
for ii=1:6
    Frecov1(:,ii)=Frecov1(:,ii)-Frecov1offset(ii);
end

%%

ftsz=15;  % font size

%%%%%%%%%%%%%%%%%%%%%%%%
% butterworth filter
% sample rate is 1 kHz, try cutoff frequency of 40 Hz
Wn = 80/1000;
FiltOrder = 4; % filter order
[den,num]=butter(FiltOrder,Wn);
Frecov1=filter(den,num,Frecov1);
% filter accelerometer as well
AX=filter(den,num,AX);
AY=filter(den,num,AY);
AZ=filter(den,num,AZ);

%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
clf;
set(gcf,'color','w');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 8 4]);

subplot(2,1,1);hold all;box on;
plot(T(:,1),Frecov1(:,1),'LineWidth',1.5)
plot(T(:,1),Frecov1(:,2),'LineWidth',1.5)
plot(T(:,1),Frecov1(:,3),'g-','LineWidth',1.5)
line([0 12.3],[0 0],'color','k');
ylabel('F (N)','fontsize',ftsz)
set(gca,'fontsize',ftsz);
legend('F_x','F_y','F_z');
xlabel('Time (s)','fontsize',ftsz);
xlim([0 maxt]);
ylim([-0.4 0.4]);

subplot(2,1,2);hold all;box on;
plot(T(:,1),Frecov1(:,4),'LineWidth',1.5)
plot(T(:,1),Frecov1(:,5),'LineWidth',1.5)
plot(T(:,1),Frecov1(:,6),'g-','LineWidth',1.5)
line([0 12.3],[0 0],'color','k');
ylabel('M (mN*m)','fontsize',ftsz)
set(gca,'fontsize',ftsz);
legend('M_x','M_y','M_z');
xlabel('Time (s)','fontsize',ftsz);
xlim([0 maxt]);
ylim([-20 20]);

%display('line 87')
%return; % for debugging
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot of robot state and force sensor
figure(2)
clf;
ha = tight_subplot(7,1,[.02 0],[.1 .08],[.1 .01]);
axes(ha(1)); plot(time(1:s:end)/1000,TorqueR(1:s:end),...
    'r-','LineWidth',2);
hold on; plot(time(1:s:end)/1000,-TorqueL(1:s:end),...
    'b:','LineWidth',2);ylabel('\tau (mN-m)','FontSize', 14, 'FontName', 'CMU Serif');axis([0,maxt,-2.1,2.1]); 
legend('Right leg','Left leg','Location','west')
axes(ha(2)); 
plot(time(1:s:end)/1000,AngleZ(1:s:end),'g','LineWidth',1);
% plot(time(1:s:end)/1000,GyroX(1:s:end),'r','LineWidth',1);
hold on; 
%plot(time(1:s:end)/1000,GyroY(1:s:end),'b','LineWidth',1);
%plot(time(1:s:end)/1000,AngleZ(1:s:end),    'g','LineWidth',1);
ylabel('\theta_z (rad)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,maxt,-1,1]); % legend('Gx','Gy', 'Gz','Location','west')
   
 %
axes(ha(3)); plot(time(1:s:end)/1000,AX(1:s:end),'k','LineWidth',2);
    ylabel('x" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([0,maxt,-25,25]);
axes(ha(4)); plot(time(1:s:end)/1000,AY(1:s:end),'k','LineWidth',2);
    ylabel('y" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([0,maxt,-25,25]);
axes(ha(5)); plot(time(1:s:end)/1000,AZ(1:s:end),'k','LineWidth',2);
    ylabel('z" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([-0,maxt,-5,30]);
%%%%% now plot contact forces and torques %%%%%%%%%%
axes(ha(6)); plot(time(1:s:end)/1000,Frecov1(1:s:end,1),'r','LineWidth',2);  
hold on; plot(time(1:s:end)/1000,Frecov1(1:s:end,2),'b','LineWidth',2); 
    plot(time(1:s:end)/1000,Frecov1(1:s:end,3),'g','LineWidth',2); 
ylabel('F(N)','FontSize', 14, 'FontName', 'CMU Serif');
    axis([0,maxt,-1.5,1.5]); legend('F_x','F_y','F_z','Location','west')
axes(ha(7)); plot(time(1:s:end)/1000,Frecov1(1:s:end,4),'r','LineWidth',2);  
hold on; plot(time(1:s:end)/1000,Frecov1(1:s:end,5),'b','LineWidth',2); 
    plot(time(1:s:end)/1000,Frecov1(1:s:end,6),'g','LineWidth',2); 
ylabel('M (mN-M)','FontSize', 14, 'FontName', 'CMU Serif');
    axis([0,maxt,-50,50]); legend('M_x','M_y','M_z','Location','west')

set(ha(1:6),'XTickLabel','') % only 1 time lable
for i = 1:7
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',14);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid OFF
    
end
axes(ha(7));
xlabel('Time (s)','FontSize', 18, 'FontName', 'CMU Serif');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 12 8]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot of robot state and force sensor - w/o accelerometer
figure(3)
clf;

ha = tight_subplot(4,1,[.02 0],[.1 .08],[.1 .01]);
axes(ha(1)); plot(time(1:s:end)/1000,TorqueR(1:s:end),...
    'r-','LineWidth',2);
hold on; plot(time(1:s:end)/1000,-TorqueL(1:s:end),...
    'b:','LineWidth',2);ylabel('\tau (mN-m)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,1.1*maxt,-2.5,2.5]); legend('Right leg','Left leg')
%rightLegPos = data(:,2)*legScale;
%leftLegPos = data(:,3)*legScale;
axes(ha(2)); plot(time(1:s:end)/1000,mod(rightLegPos(1:s:end), 2*pi),...
    'r-','LineWidth',2);
hold on; plot(time(1:s:end)/1000,mod(leftLegPos(1:s:end), 2*pi),...
    'b:','LineWidth',2);
ylabel('leg position (rad)','FontSize', 14, 'FontName', 'CMU Serif');
%axes(ha(2)); plot(time(1:s:end)/1000,AngleZ(1:s:end),...
%    'k','LineWidth',3);ylabel('\theta_Z (rad)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,1.1*maxt,0,2*pi]);% gyro Z angle
legend('Right leg','Left leg')
% axes(ha(3)); plot(time(1:s:end)/1000,AX(1:s:end),'k','LineWidth',2);
%     ylabel('x" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([0,maxt,-15,15]);
% axes(ha(4)); plot(time(1:s:end)/1000,AY(1:s:end),'k','LineWidth',2);
%     ylabel('y" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([0,maxt,-15,15]);
% axes(ha(5)); plot(time(1:s:end)/1000,AZ(1:s:end),'k','LineWidth',2);
%     ylabel('z" (m/s^2)','FontSize', 14, 'FontName', 'CMU Serif');axis([-0,maxt,-5,25]);
%%%%% now plot contact forces and torques %%%%%%%%%%
axes(ha(3)); plot(time(1:s:end)/1000,Frecov1(1:s:end,1),'r','LineWidth',2);  
hold on; plot(time(1:s:end)/1000,Frecov1(1:s:end,2),'b','LineWidth',2); 
    plot(time(1:s:end)/1000,Frecov1(1:s:end,3),'g','LineWidth',2); 
ylabel('F(N)','FontSize', 14, 'FontName', 'CMU Serif');
    axis([0,1.1*maxt,-1,1]); legend('F_x','F_y','F_z')
axes(ha(4)); plot(time(1:s:end)/1000,Frecov1(1:s:end,4),'r','LineWidth',2);  
hold on; plot(time(1:s:end)/1000,Frecov1(1:s:end,5),'b','LineWidth',2); 
    plot(time(1:s:end)/1000,Frecov1(1:s:end,6),'g','LineWidth',2); 
ylabel('M (mN-M)','FontSize', 14, 'FontName', 'CMU Serif');
    axis([0,1.1*maxt,-30,30]); legend('M_x','M_y','M_z')

set(ha(1:3),'XTickLabel','') % only 1 time label
for i = 1:4
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',14);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid OFF
    
end
axes(ha(4));
xlabel('Time (s)','FontSize', 18, 'FontName', 'CMU Serif');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 10 12]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot of leg position versus force
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frecov1(n,1) = Fx, (n,2)=Fy, (n,3)=Fz
figure(4)
clf;
s=1;
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 10 6]);
ha = tight_subplot(3,2,[.02 0.02],[.1 .08],[.1 .03]);
% ha = tight_subplot(columns, rows, [gapy gapx], marg_h, marg_w)

axes(ha(1)); plot(mod(rightLegPos(1:s:end/2),2*pi),TorqueR(1:s:end/2),...
    'ro','LineWidth',1);
hold on; plot(mod(leftLegPos(1:s:end/2),2*pi),-TorqueL(1:s:end/2),...
    'b+','LineWidth',1);
ylabel('\tau (mN-m)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,1.1*maxt,-2.5,2.5]); 
title('Before contact','FontSize', 12, 'FontName', 'CMU Serif');

axes(ha(2)); plot(mod(rightLegPos(end/2:s:end),2*pi),TorqueR(end/2:s:end),...
    'ro','LineWidth',1);
hold on; plot(mod(leftLegPos(end/2:s:end),2*pi),-TorqueL(end/2:s:end),...
    'b+','LineWidth',1);
set(ha(2),'YTickLabel','') % only label on left plot
axis([0,1.1*maxt,-2.5,2.5]); legend('Right torque','Left torque')
title('after contact','FontSize', 12, 'FontName', 'CMU Serif');
%
%%%%%%%% body forces before and after contact with walls %%%%%%%%%%
axes(ha(3));  plot(mod(leftLegPos(1:s:end/2),2*pi),Frecov1(1:s:end/2,2),'r+','LineWidth',1);  
hold on; 
ylabel('F(N)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,1.1*maxt,-1.5,1.5])
% plot in wall region separately
axes(ha(4));
plot(mod(leftLegPos(end/2:s:end),2*pi),Frecov1(end/2:s:end,2),'r+','LineWidth',1); 
set(ha(4),'YTickLabel','') % only label on left plot
axis([0,1.1*maxt,-1.5,1.5]); legend('F_y')



% moment measure, Mx
axes(ha(5)); plot(mod(leftLegPos(1:s:end/2),2*pi),Frecov1(1:s:end/2,4),'ro','LineWidth',1); 
ylabel('M (mN-M)','FontSize', 14, 'FontName', 'CMU Serif');
    axis([0,1.1*maxt,-40,40]); 
% 
hold on; 
% plot(time(1:s:end)/1000,Frecov1(1:s:end,5),'b+','LineWidth',1); 
% 6=Mz
axes(ha(6))
    plot(mod(leftLegPos(end/2:s:end),2*pi),Frecov1(end/2:s:end,6),'g*','LineWidth',1);
    axis([0,1.1*maxt,-40,40])
    set(ha(6),'YTickLabel','') % only label on left plot
legend('M_x')
set(ha(1:4),'XTickLabel','') % only 1 time label
for i = 1:3
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',14);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid OFF
    
end
axes(ha(5));
xlabel('Leg position (rad)','FontSize', 18, 'FontName', 'CMU Serif');
axes(ha(6));
xlabel('Leg position (rad)','FontSize', 18, 'FontName', 'CMU Serif');

%%%%%%%%%%%%%%%%%%%
% now plot data on a cycle by cycle basis
cycle_plot

