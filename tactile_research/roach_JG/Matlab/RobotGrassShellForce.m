%%%%% original version written by Josh Goldberg My 2015
clear;
close all;

processflag=3;
% all data=1
% vary layer=2;
% vary spacing=3
% vary stiffness & frequency=4

%%%  various options for code below- should use smaller functions instead
plotting=1;
makemovie=0;  %% makes a .jpg plot for each time step
plottrials=1;
filtering=0;
filterfreq=100;

layerind=0:5;
spacingind=9:12;
stiffnessind=[4 6];
freqind=7:3:13;

behavdef{1}='pitchwalk';
behavdef{2}='pitchasym';
behavdef{3}='pitchsym';
behavdef{4}='pitchbetween';
behavdef{5}='roll';
behavdef{6}='rollstuck';
behavdef{7}='stuckbetween';
behavdef{8}='';
behavdef{9}='walk';

%dirname1='D:\Dropbox\Work\Grass\working\robot shell force\';
dirname1='..';

load(strcat(dirname1,'\telemetry\N_matrix_trial9.mat'));
pathmov=strcat(dirname1,'\videos\plots\');

if processflag==1    
    %load(strcat(dirname1,'\telemetry\all_data.mat'));
    %dirname=strcat(dirname1,'\telemetry\alldata\');
    load(strcat(dirname1,'\telemetry\testdata.mat'));
    dirname=strcat(dirname1,'\telemetry\testdata\');
else if processflag==2
        load(strcat(dirname1,'\telemetry\vary_layer.mat'));
        dirname=strcat(dirname1,'\telemetry\vary layer\');
    else if processflag==3
            load(strcat(dirname1,'\telemetry\vary_spacing.mat'));
            dirname=strcat(dirname1,'\telemetry\vary spacing\');
        else
            load(strcat(dirname1,'\telemetry\vary_stiffness_frequency.mat'));
            dirname=strcat(dirname1,'\telemetry\vary stiffness & frequency\');
        end
    end
end

% cd (dirname);

ply=bb(:,1);
spacing=bb(:,2);
height=bb(:,3);
layer=bb(:,4);
freq=bb(:,5);
trial=bb(:,6);
start0=bb(:,7);
end0=bb(:,8);
start1=bb(:,9);
end1=bb(:,10);
start2=bb(:,11);
end2=bb(:,12);
avgnum=bb(:,13);
behav1=bb(:,14);
behav2=bb(:,15);
touchgrass=bb(:,16);
telembad=bb(:,17);

start1(find(start1==0))=start0(find(start1==0));
end1(find(end1==0))=end0(find(end1==0));
start1(find(avgnum==0))=NaN;
end1(find(avgnum==0))=NaN;
start1(find(telembad==1))=NaN;
end1(find(telembad==1))=NaN;

%%

filenamelist=0;clear filenamelist;

ddtop=dir(dirname);

count=1;

for jj=3:size(ddtop,1);
    
    filename=ddtop(jj).name;
    filenamenoext=filename;
    filenamenoext(length(filename)-3:length(filename))=[];

    filenamelist{count}=filename;    
    filenamenoextlist{count}=filenamenoext;
    
    count=count+1;
end

filenamelist=filenamelist';
lff=length(filenamelist);

%%

avgFFx1=zeros(lff,1);
avgFFy1=zeros(lff,1);
avgFFz1=zeros(lff,1);
avgNNx1=zeros(lff,1);
avgNNy1=zeros(lff,1);
avgNNz1=zeros(lff,1);

avgFFx2=zeros(lff,1);
avgFFy2=zeros(lff,1);
avgFFz2=zeros(lff,1);
avgNNx2=zeros(lff,1);
avgNNy2=zeros(lff,1);
avgNNz2=zeros(lff,1);
%return % for debugging
cd(dirname);  % change to directory where all data files are stored
for jj=1:lff;
    
    namein=char(filenamelist(jj));
    nameinnoext=char(filenamenoextlist(jj));
    disp(namein);
    load(namein);
   
    T = csvread(namein,9,0);

    S = T(:,17:24);
    A = [S(:,1),S(:,1).^2,S(:,1).^3,S(:,2),S(:,2).^2,S(:,2).^3,S(:,3),S(:,3).^2,S(:,3).^3,S(:,4),S(:,4).^2,S(:,4).^3,S(:,5),S(:,5).^2,S(:,5).^3,S(:,6),S(:,6).^2,S(:,6).^3,S(:,7),S(:,7).^2,S(:,7).^3,S(:,8),S(:,8).^2,S(:,8).^3];
    Frecov = A*N;
    Frecov1 = Frecov;
    i = 2;

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

    time=T(:,1)/1000000;
    Frecov1offset=mean(Frecov1(1:50,:));
    for ii=1:6
        Frecov1(:,ii)=Frecov1(:,ii)-Frecov1offset(ii);
    end
    
    timeend(jj)=max(time);
    indend(jj)=fix((timeend(jj))*30);
    
    if filtering==1    
        Frecov1=butterfilter(Frecov1,filterfreq,5);
    end

    Fx=Frecov1(:,1);
    Fy=Frecov1(:,2);
    Fz=Frecov1(:,3);
    Nx=Frecov1(:,4);
    Ny=Frecov1(:,5);
    Nz=Frecov1(:,6);
    
    samplingfreq=30;
    runlength=timeend(jj);
    timez=0:(1/samplingfreq):runlength;

    windowwidth=1;
    avgwidth=1;
    movstep=1;
    lnwz=0.5;
    lnwz2=2;
    ftsz=10;
    ftsz2=15;
    mksz=10;
    mksz2=20;
    
    FFx=interp1(time,Fx,timez,'linear');
    FFy=interp1(time,Fy,timez,'linear');
    FFz=interp1(time,Fz,timez,'linear');
    NNx=interp1(time,Nx,timez,'linear');
    NNy=interp1(time,Ny,timez,'linear');
    NNz=interp1(time,Nz,timez,'linear');
    
    if processflag>2
        if touchgrass==-1
           FFy=-FFy; 
           NNx=-NNx; 
           NNz=-NNz; 
        end
    end
        
    if avgnum(jj)==1
        
        if telembad(jj)==0
        
            avgFFx1(jj)=mean(FFx(start1(jj):end1(jj)));
            avgFFy1(jj)=mean(FFy(start1(jj):end1(jj)));
            avgFFz1(jj)=mean(FFz(start1(jj):end1(jj)));
            avgNNx1(jj)=mean(NNx(start1(jj):end1(jj)));
            avgNNy1(jj)=mean(NNy(start1(jj):end1(jj)));
            avgNNz1(jj)=mean(NNz(start1(jj):end1(jj)));

            avgFFx2(jj)=NaN;
            avgFFy2(jj)=NaN;
            avgFFz2(jj)=NaN;
            avgNNx2(jj)=NaN;
            avgNNy2(jj)=NaN;
            avgNNz2(jj)=NaN;
        
        end
        
    else if avgnum(jj)==2
            avgFFx1(jj)=mean(FFx(start1(jj):end1(jj)));
            avgFFy1(jj)=mean(FFy(start1(jj):end1(jj)));
            avgFFz1(jj)=mean(FFz(start1(jj):end1(jj)));
            avgNNx1(jj)=mean(NNx(start1(jj):end1(jj)));
            avgNNy1(jj)=mean(NNy(start1(jj):end1(jj)));
            avgNNz1(jj)=mean(NNz(start1(jj):end1(jj)));
        
            avgFFx2(jj)=mean(FFx(start2(jj):end2(jj)));
            avgFFy2(jj)=mean(FFy(start2(jj):end2(jj)));
            avgFFz2(jj)=mean(FFz(start2(jj):end2(jj)));
            avgNNx2(jj)=mean(NNx(start2(jj):end2(jj)));
            avgNNy2(jj)=mean(NNy(start2(jj):end2(jj)));
            avgNNz2(jj)=mean(NNz(start2(jj):end2(jj)));      
            
        else if avgnum(jj)==0
                
            avgFFx1(jj)=NaN;
            avgFFy1(jj)=NaN;
            avgFFz1(jj)=NaN;
            avgNNx1(jj)=NaN;
            avgNNy1(jj)=NaN;
            avgNNz1(jj)=NaN;
            
            avgFFx2(jj)=NaN;
            avgFFy2(jj)=NaN;
            avgFFz2(jj)=NaN;
            avgNNx2(jj)=NaN;
            avgNNy2(jj)=NaN;
            avgNNz2(jj)=NaN;
            
            end
        end
    end

    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if plotting==1
        
        figure(1);clf;

        set(gcf,'color','w');
        set(gcf,'Units','inches');
        set(gcf,'Position',[1 1 8 4]);

        subplot(2,1,1);hold all;box on;
        plot(timez,FFx,'b-','linewidth',lnwz);
        plot(timez,FFy,'r-','linewidth',lnwz);
        plot(timez,FFz,'g-','linewidth',lnwz);
        line([start1(jj) end1(jj)]/30,[avgFFx1(jj) avgFFx1(jj)],'color','b','linewidth',lnwz2);
        line([start1(jj) end1(jj)]/30,[avgFFy1(jj) avgFFy1(jj)],'color','r','linewidth',lnwz2);
        line([start1(jj) end1(jj)]/30,[avgFFz1(jj) avgFFz1(jj)],'color','g','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgFFx2(jj) avgFFx2(jj)],'color','b','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgFFy2(jj) avgFFy2(jj)],'color','r','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgFFz2(jj) avgFFz2(jj)],'color','g','linewidth',lnwz2);
        line([start1(jj) start1(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
        line([end1(jj) end1(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
        line([start2(jj) start2(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
        line([end2(jj) end2(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
        line([0 runlength],[0 0],'linestyle',':','color','k');
        ylabel('Force (N)','fontsize',ftsz)
        set(gca,'fontsize',ftsz);
        legend('Forward','Left','Upward','location','southeast');
        xlabel('Time (s)','fontsize',ftsz);
        xlim([0 runlength*1.2]);
        ylim([min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])]);
        title(nameinnoext,'Interpreter','none');

        subplot(2,1,2);hold all;box on;
        plot(timez,NNx,'b-','linewidth',lnwz);
        plot(timez,NNy,'r-','linewidth',lnwz);
        plot(timez,NNz,'g-','linewidth',lnwz);
        line([start1(jj) end1(jj)]/30,[avgNNx1(jj) avgNNx1(jj)],'color','b','linewidth',lnwz2);
        line([start1(jj) end1(jj)]/30,[avgNNy1(jj) avgNNy1(jj)],'color','r','linewidth',lnwz2);
        line([start1(jj) end1(jj)]/30,[avgNNz1(jj) avgNNz1(jj)],'color','g','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgNNx2(jj) avgNNx2(jj)],'color','b','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgNNy2(jj) avgNNy2(jj)],'color','r','linewidth',lnwz2);
        line([start2(jj) end2(jj)]/30,[avgNNz2(jj) avgNNz2(jj)],'color','g','linewidth',lnwz2);
        line([start1(jj) start1(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
        line([end1(jj) end1(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
        line([start2(jj) start2(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
        line([end2(jj) end2(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
        line([0 runlength],[0 0],'linestyle',':','color','k');
        ylabel('Torque (mN*m)','fontsize',ftsz)
        set(gca,'fontsize',ftsz);
        legend('Roll right','Pitch down','Yaw left','location','southeast');
        xlabel('Time (s)','fontsize',ftsz);
        xlim([0 runlength*1.2]);
        ylim([min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])]);
        disp('Next?')
        pause;
        
    end

   % return  % for debugging
    %%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if makemovie==1        
     makemoviefiles
    end
end
indend=indend';
disp('Done Plotting.')
%%
%return % for debugging
dx=0.1;dy=0;

figure(5);clf;

subplot(2,1,1);hold all;
plot(1:lff,avgFFx1,'bo');
plot(1:lff,avgFFy1,'ro');
plot(1:lff,avgFFz1,'go');

plot(1:lff,avgFFx2,'bo','markerfacecolor','b');
plot(1:lff,avgFFy2,'ro','markerfacecolor','r');
plot(1:lff,avgFFz2,'go','markerfacecolor','g');

for jj=1:lff
    text(jj+dx,avgFFx1(jj)+dy,behavdef{behav1(jj)});
    if behav2(jj)~=0
        text(jj+dx,avgFFx2(jj)+dy,behavdef{behav2(jj)});
    end
end

legend('Forward','Left','Upward','location','northwest');

subplot(2,1,2);hold all;
plot(1:lff,avgNNx1,'bo');
plot(1:lff,avgNNy1,'ro');
plot(1:lff,avgNNz1,'go');

plot(1:lff,avgNNx2,'bo','markerfacecolor','b');
plot(1:lff,avgNNy2,'ro','markerfacecolor','r');
plot(1:lff,avgNNz2,'go','markerfacecolor','g');

for jj=1:lff
    text(jj+dx,avgNNx1(jj)+dy,behavdef{behav1(jj)});
    if behav2(jj)~=0
        text(jj+dx,avgNNx2(jj)+dy,behavdef{behav2(jj)});
    end
end

legend('Roll right','Pitch down','Yaw left','location','southwest');

%%

for ii=1:lff/3
    
    meanind=(ii-1)*3+1:(ii-1)*3+3;
   
    avgFFx1mean(ii)=mean(avgFFx1(meanind));
    avgFFy1mean(ii)=mean(avgFFy1(meanind));
    avgFFz1mean(ii)=mean(avgFFz1(meanind));
    avgNNx1mean(ii)=mean(avgNNx1(meanind));
    avgNNy1mean(ii)=mean(avgNNy1(meanind));
    avgNNz1mean(ii)=mean(avgNNz1(meanind));
    
    avgFFx1std(ii)=std(avgFFx1(meanind));
    avgFFy1std(ii)=std(avgFFy1(meanind));
    avgFFz1std(ii)=std(avgFFz1(meanind));
    avgNNx1std(ii)=std(avgNNx1(meanind));
    avgNNy1std(ii)=std(avgNNy1(meanind));
    avgNNz1std(ii)=std(avgNNz1(meanind));
    
    avgFFx2mean(ii)=mean(avgFFx2(meanind));
    avgFFy2mean(ii)=mean(avgFFy2(meanind));
    avgFFz2mean(ii)=mean(avgFFz2(meanind));
    avgNNx2mean(ii)=mean(avgNNx2(meanind));
    avgNNy2mean(ii)=mean(avgNNy2(meanind));
    avgNNz2mean(ii)=mean(avgNNz2(meanind));
    
    avgFFx2std(ii)=std(avgFFx2(meanind));
    avgFFy2std(ii)=std(avgFFy2(meanind));
    avgFFz2std(ii)=std(avgFFz2(meanind));
    avgNNx2std(ii)=std(avgNNx2(meanind));
    avgNNy2std(ii)=std(avgNNy2(meanind));
    avgNNz2std(ii)=std(avgNNz2(meanind));
    
end

%%
% plot aggregate forces/moments in various terrain types
cd('../../Matlab')
%maketerrainplot

