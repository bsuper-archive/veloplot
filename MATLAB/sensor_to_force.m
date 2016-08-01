%%
close all
clear all

F = csvread('/Users/briansu/biomimetics/jgoldberg_computer_dump/Dropbox_data/trial5_force.csv',0,1);
F = F(:,3:8);
S = csvread('/Users/briansu/biomimetics/jgoldberg_computer_dump/Dropbox_data/trial5_tactile.csv');
S = S(:,2:9);

subplot(2,1,1)
plot(F(28366:30000,1))
subplot(2,1,2)
plot(S(5315:6000,1))
%close all

Fthresh = .008;
F1 = find(F(:,1)>Fthresh,1);
Foffset = 28000; %78700; %trial4
F2 = Foffset+find(F(1+Foffset:end,1)<-Fthresh,1);
F3 = find(F(:,1)>Fthresh);

Sthresh = 1460;
S1 = find(S(:,1)>Sthresh,1);
Soffset = 5200; %14900; %trial4
S2 = Soffset+find(S(1+Soffset:end,1)>Sthresh,1);
S3 = find(S(:,1)>Sthresh);

close all
%subplot(2,1,1)
%plot(F(F3(end)-15000:end,1))
%subplot(2,1,2)
%plot(S(S3(end)-400:end,1))

subplot(2,2,1)
plot(F(1:F1+500,1))
subplot(2,2,3)
plot(S(1:S1+100,1))

subplot(2,2,2)
plot(F(F2-500:end,1))
subplot(2,2,4)
plot(S(S2-100:end,1))

%close all
%subplot(2,1,1)
%plot(F(:,1))
%subplot(2,1,2)
%plot(S(:,1))

Fdelta=F2-F1;
Sdelta=S2-S1;
Fsamp=resample(F,Sdelta,Fdelta);
close all
subplot(2,1,1)
plot(Fsamp)
subplot(2,1,2)
plot(S)



Fthresh = .008;
F1samp = find(Fsamp(:,1)>Fthresh,1);

FSdelta = F1samp-S1;

if FSdelta >= 0
    F = Fsamp(FSdelta+1:end,:);
else
    F = Fsamp;
    S = S(-FSdelta+1:end,:);
end

close all
subplot(2,1,1)
plot(F)
subplot(2,1,2)
plot(S)


if size(F,1) >= size(S,1)
    F = F(1:size(S,1),:);
else
    S = S(1:size(F,1),:);
end

close all
subplot(2,1,1)
plot(F(:,:))
subplot(2,1,2)
plot(S(:,:))

% moving average filter
filter_len = 100;
F_avg = zeros(size(F,1)-filter_len+1,size(F,2));
for col = 1:size(F,2)
    for row = 1:size(F,1)-filter_len+1
        F_avg(row,col) = mean(F(row:row+filter_len-1,col))-mean(F(1:100,col));
    end
end
%filter_len = filter_len
S_avg = zeros(size(S,1)-filter_len+1,size(S,2));
for col = 1:size(S,2)
    for row = 1:size(S,1)-filter_len+1
        S_avg(row,col) = mean(S(row:row+filter_len-1,col))-mean(S(1:100,col));
    end
end

close all
subplot(2,1,1)
plot(F_avg(:,:))
subplot(2,1,2)
plot(S_avg(:,:))
F=F_avg(800:end,:);
S=S_avg(800:end,:);%(1:size(F,1),:);

%
%close all

%A = [1./(S(:,1).^2),1./(S(:,1)),ones(size(S,1),1),S(:,1),S(:,1).^2,1./(S(:,2).^2),1./(S(:,2)),ones(size(S,1),1),S(:,2),S(:,2).^2,1./(S(:,3).^2),1./(S(:,3)),ones(size(S,1),1),S(:,3),S(:,3).^2,1./(S(:,4).^2),1./(S(:,4)),ones(size(S,1),1),S(:,4),S(:,4).^2,1./(S(:,5).^2),1./(S(:,5)),ones(size(S,1),1),S(:,5),S(:,5).^2,1./(S(:,6).^2),1./(S(:,6)),ones(size(S,1),1),S(:,6),S(:,6).^2,1./(S(:,7).^2),1./(S(:,7)),ones(size(S,1),1),S(:,7),S(:,7).^2,1./(S(:,8).^2),1./(S(:,8)),ones(size(S,1),1),S(:,8),S(:,8).^2];
%A = [1./(S(:,1).^2),1./(S(:,1)),ones(size(S,1),1),S(:,1),S(:,1).^2,1./(S(:,2).^2),1./(S(:,2)),S(:,2),S(:,2).^2,1./(S(:,3).^2),1./(S(:,3)),S(:,3),S(:,3).^2,1./(S(:,4).^2),1./(S(:,4)),S(:,4),S(:,4).^2,1./(S(:,5).^2),1./(S(:,5)),S(:,5),S(:,5).^2,1./(S(:,6).^2),1./(S(:,6)),S(:,6),S(:,6).^2,1./(S(:,7).^2),1./(S(:,7)),S(:,7),S(:,7).^2,1./(S(:,8).^2),1./(S(:,8)),S(:,8),S(:,8).^2];
%A = [ones(size(S,1),1),S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
%A = [ones(size(S,1),1),S(:,1),S(:,1).^2,S(:,1).^3,S(:,1).^4,S(:,2),S(:,2).^2,S(:,2).^3,S(:,2).^4,S(:,3),S(:,3).^2,S(:,3).^3,S(:,3).^4,S(:,4),S(:,4).^2,S(:,4).^3,S(:,4).^4,S(:,5),S(:,5).^2,S(:,5).^3,S(:,5).^4,S(:,6),S(:,6).^2,S(:,6).^3,S(:,6).^4,S(:,7),S(:,7).^2,S(:,7).^3,S(:,7).^4,S(:,8),S(:,8).^2,S(:,8).^3,S(:,8).^4];

A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
%A = [S(:,1),S(:,1).^2,S(:,1).^-1,S(:,2),S(:,2).^2,S(:,2).^-1,S(:,3),S(:,3).^2,S(:,3).^-1,S(:,4),S(:,4).^2,S(:,4).^-1,S(:,5),S(:,5).^2,S(:,5).^-1,S(:,6),S(:,6).^2,S(:,6).^-1,S(:,7),S(:,7).^2,S(:,7).^-1,S(:,8),S(:,8).^2,S(:,8).^-1];

N = A\F;

%F=F_avg(1:end,:);
%S=S_avg(1:end,:);%(1:size(F,1),:);
%A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
Frecov = A*N;
%%
close all
subplot(3,1,1)
hold on
f = 2;
plot(F(:,f))%./-min(F(:,f)))
plot(Frecov(:,f))%./-min(Frecov(:,f)))
%axis([0,6000,-2,2])
subplot(3,1,2)
plot(S(:,:))
%axis([0,6000,0,5000])
subplot(3,1,3)
plot(abs((F(:,f)-Frecov(:,f))./F(:,f)))%/-min(F(:,f)-Frecov(:,f)))
axis([0,12000,-1,1])
%close all
mean_abs_error = mean(abs(F-Frecov))
mean_percent_error = mean(abs((F-Frecov)./F))
Ferror = F-Frecov;
test = F-Frecov;
for j = 1:size(F,2)
    for i = 1:size(F,1)
        test(i,j) = test(i,j)/F(i,j);
    end
end
mean(abs(test))
median(abs(test))
%%
bound = 1.5;
close all
subplot(2,3,1)
plot(F(:,1),Frecov(:,1),'.')
xlabel('Fx Measured (N)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('Fx Recovered (N)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,bound],'YLim',[-bound,bound]);

subplot(2,3,2)
plot(F(:,2),Frecov(:,2),'.')
xlabel('Fy Measured (N)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('Fy Recovered (N)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,bound],'YLim',[-bound,bound]);

subplot(2,3,3)
plot(F(:,3),Frecov(:,3),'.')
xlabel('Fz Measured (N)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('Fz Recovered (N)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,0.1],'YLim',[-bound,0.1]);

bound = 60;
subplot(2,3,4)
plot(F(:,4),Frecov(:,4),'.')
xlabel('Mx Measured (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('Mx Recovered (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,bound],'YLim',[-bound,bound]);

bound = 80;
subplot(2,3,5)
plot(F(:,5),Frecov(:,5),'.')
xlabel('My Measured (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('My Recovered (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,bound],'YLim',[-bound,bound]);

bound = 15;
subplot(2,3,6)
plot(F(:,6),Frecov(:,6),'.')
xlabel('Mz Measured (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
ylabel('Mz Recovered (mN-m)','FontSize', 24, 'FontName', 'CMU Serif');
set(gca,'FontName','CMU Serif','FontSize',18,'XLim',[-bound,bound],'YLim',[-bound,bound]);


set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
pause(1);
export_fig '/Users/jgoldberg/Downloads/josh_ms/figures/recovered_vs_measured' '-pdf' '-transparent'


%% FORCES only measure mean percent error on non zero samples
close all
clc
for i = 1:3
    Ftest = F(:,i);
    inds = abs(Ftest)>0.1;
    Fvals = Ftest(inds);
    Ftestrecov = Frecov(:,i);
    Fvalsrecov = Ftestrecov(inds);
    subplot(3,1,i)
    hold on
    Ftesterr = abs((Fvals-Fvalsrecov)./Fvals);
    plot(Fvals)
    plot(Fvalsrecov)
    %plot(Ftesterr)
    m=mean(Ftesterr)
    %s=std(Ftesterr)
end

%% MOMENTS only measure mean percent error on non zero samples
close all

for i = 4:6
    Ftest = F(:,i);
    inds = abs(Ftest)>5;
    if i == 6
        %inds = abs(Ftest)>2;
    end
    Fvals = Ftest(inds);
    Ftestrecov = Frecov(:,i);
    Fvalsrecov = Ftestrecov(inds);
    subplot(3,1,i-3)
    hold on
    Ftesterr = abs((Fvals-Fvalsrecov)./Fvals);
    plot(Fvals)
    plot(Fvalsrecov)
    %plot(Ftesterr)
    m = mean(Ftesterr)
    %s = std(Ftesterr)
end


%%
clc
close all

% Change default text fonts.
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultTextFontSize', 30)

subplot(1,2,1)
h1 = boxplot(Ferror(:,1:3)*1000,'sym','k+', 'colors','k','labels',{'Fx','Fy','Fz'})
ylabel('Force calibration error (mN)','FontSize', 30, 'FontName', 'CMU Serif')

subplot(1,2,2)
h2 = boxplot(Ferror(:,4:end),'sym','k+', 'colors','k','labels',{'Mx','My','Mz'})
ylabel('Torque calibration error (mN-m)','FontSize', 30, 'FontName', 'CMU Serif')

set(get(get(h1(1),'parent'),'Parent'),'FontSize',24,'FontName', 'CMU Serif')
set(get(get(h2(1),'parent'),'Parent'),'FontSize',24,'FontName', 'CMU Serif')
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
pause(1);
%export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/calibration_error_boxplot' '-pdf' '-transparent'

%%
close all
% Change default text fonts.
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultTextFontSize', 30)

ha = tight_subplot(3,2,[.01 .1],[.1 .08],[.1 .01]);

axes(ha(1));
hist1 = histogram(Ferror(:,1)*1000,'Normalization','probability');
set(hist1,'FaceColor','k')
%axis([-.1,.1,0,2000])
legend('Fx')
title(' ')

axes(ha(3));
hist2 = histogram(Ferror(:,2)*1000,'Normalization','probability');
set(hist2,'FaceColor','k')
%axis([-.1,.1,0,1500])
legend('Fy')

axes(ha(5));
hist3 = histogram(Ferror(:,3)*1000,'Normalization','probability');
set(hist3,'FaceColor','k')
%axis([-.1,.1,0,2000])
legend('Fz')

xlabel('Error (mN)','FontSize', 32, 'FontName', 'CMU Serif')

axes(ha(2));
hist4 = histogram(Ferror(:,4),'Normalization','probability');
set(hist4,'FaceColor','k')
%axis([-3,3,0,1000])
legend('Mx')

axes(ha(4));
hist5 = histogram(Ferror(:,5),'Normalization','probability');
set(hist5,'FaceColor','k')
%axis([-3,3,0,2000])
legend('My')

axes(ha(6));
hist6 = histogram(Ferror(:,6),'Normalization','probability');
set(hist6,'FaceColor','k')
%axis([-3,3,0,1])
legend('Mz')

xlabel('Error (mN-m)','FontSize', 32, 'FontName', 'CMU Serif')

for i = 1:6
    temp = get(ha(i),'YTick')
    set(ha(i),'YTick',temp(1:end-1))
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',30);
end

set(ha([1,3,5]),'XLim',[-100,100])
set(ha([2,4,6]),'XLim',[-5,5])
set(ha(1:4),'XTickLabel','');
set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
pause(1);
export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/calibration_error_histogram' '-pdf' '-transparent'


%%
close all
% Change default text fonts.
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultTextFontSize', 30)

subplot(3,2,1)
hold on
plot(F(:,1));
plot(Frecov(:,1));
ylabel('Fx (N)')
xlabel('Time')
legend('Measured','Reconstructed')


subplot(3,2,3)
hold on
plot(F(:,2));
plot(Frecov(:,2));
ylabel('Fy (N)')
xlabel('Time')
legend('Measured','Reconstructed')

subplot(3,2,5)
hold on
plot(F(:,3));
plot(Frecov(:,3));
ylabel('Fz (N)')
xlabel('Time')
legend('Measured','Reconstructed')

subplot(3,2,2)
hold on
plot(F(:,4));
plot(Frecov(:,4));
ylabel('Mx (mN*m)')
xlabel('Time')
legend('Measured','Reconstructed')

subplot(3,2,4)
hold on
plot(F(:,5));
plot(Frecov(:,5));
ylabel('My (mN*m)')
xlabel('Time')
legend('Measured','Reconstructed')

subplot(3,2,6)
hold on
plot(F(:,6));
plot(Frecov(:,6));
ylabel('Mz (mN*m)')
xlabel('Time')
legend('Measured','Reconstructed')

set(gcf,'Units','inches');
set(gcf,'Position',[1 1 14 16]);
pause(1);
%export_fig '/Users/jgoldberg/Downloads/josh_iros/figures/measured_and_reconstructed_force_torque' '-pdf' '-transparent'
