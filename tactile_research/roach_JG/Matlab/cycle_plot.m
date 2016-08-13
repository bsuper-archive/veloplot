% cycle_plot
% plot telemetry and force sensor data on cycle by cycle basis
% global variables:
% rightLegPos()
% assume lead in of some number of samples
figure(1);
clf;
hold on;
minThresh = 0.05;   % left angle is decreasing and cycling to 2 pi
maxThresh = 6.20;   % if threshold is too tight will get extra cycle
maxFound = false;
length = size(rightLegPos,1);
j = 1;
run_begin = 200;  % data from before robot starts running
xtickmin(j)=run_begin; % first sample
xtickmax(j)=run_begin;
% find each segment, keep monotonically increasing, truncate end of segment
% assume no noise beyond threshold in leg position
for i = run_begin:length
    if ((mod(rightLegPos(i), 2*pi) < minThresh) & maxFound)
        maxFound = false;
        j = j+1;
        xtickmin(j) = i;
    end
    if ((mod(rightLegPos(i),2*pi) > maxThresh) & not(maxFound))
        maxFound = true; % only found peak once
        xtickmax(j)=i;
       % str = sprintf('New cycle at i=%d, position =%8.3f', i, mod(rightLegPos(i),2*pi));
       % display(str);
        Fx=Frecov1(xtickmin(j):xtickmax(j),1);
        Fy=Frecov1(xtickmin(j):xtickmax(j),2);
        Fz=Frecov1(xtickmin(j):xtickmax(j),3);
        Fmag = sqrt(Fx.^2+Fy.^2+Fz.^2);
        plot(mod(rightLegPos(xtickmin(j):xtickmax(j)),2*pi),...
            Fmag,'r-','LineWidth',1); 
     end
end

%%%% have found all cycles, now need to pick segments
%%%% pick segments manually for now
seg1=5; seg2=14; % free, non contact segment
seg3=19; seg4=31;

%set(ha(4),'YTickLabel','') % only label on left plot
axis([0,1.1*maxt,-1.5,1.5]); legend('F_y')
ylabel('F(N)','FontSize', 14, 'FontName', 'CMU Serif');
xlabel('Leg position (rad)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,2*pi,-1.5,1.5]); legend('F_{mag}');
%%%%%%%%%%%%%%%%%%%%%

figure(4);
clf;
ha = tight_subplot(3,2,[.02 0.02],[.1 .08],[.1 .03]);
% ha = tight_subplot(columns, rows, [gapy gapx], marg_h, marg_w)

%%%%%%%
% plot each leg cycle
axes(ha(3)); plot([0 2*pi],[0 0]);
hold on; 
%for i=1:j-1
for k=seg1:seg2
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),Frecov1(xtickmin(k):xtickmax(k),2),'b-','LineWidth',1); 
  %str = sprintf('k = %d, xtickmin= %d   xtickmax=%d\n',k,xtickmin(k),xtickmax(k));
  %display(str);
end
ylabel('F(N)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,2*pi,-1.5,1.5])

axes(ha(4)); hold on; 
for k=seg3:seg4
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),Frecov1(xtickmin(k):xtickmax(k),2),'b-','LineWidth',1); 
  %str = sprintf('k = %d, xtickmin= %d   xtickmax=%d\n',k,xtickmin(k),xtickmax(k));
  %display(str);
end
set(ha(4),'YTickLabel','') % only label on left plot
axis([0,2*pi,-1.5,1.5]); legend('F_y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% leg torque   %%%%%%%%%

axes(ha(1)); plot([0 2*pi],[0 0]); hold on; 
for k=seg1:seg2
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),TorqueR(xtickmin(k):xtickmax(k)),...
    'r-','LineWidth',1);
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),-TorqueL(xtickmin(k):xtickmax(k)),...
    'b-','LineWidth',1);
end
ylabel('\tau (mN-m)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,2*pi,-2.5,2.5]); 
title('Before contact','FontSize', 12, 'FontName', 'CMU Serif');

axes(ha(2)); plot([0 2*pi],[0 0]); hold on; 
for k=seg3:seg4
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),TorqueR(xtickmin(k):xtickmax(k)),...
    'r-','LineWidth',1);
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),-TorqueL(xtickmin(k):xtickmax(k)),...
    'b-','LineWidth',1);
end
set(ha(2),'YTickLabel','') % only label on left plot
axis([0,2*pi,-2.5,2.5]); 
legend('Right torque','Left torque','Location','north')
title('after contact','FontSize', 12, 'FontName', 'CMU Serif');
%
%%%%%%%% body forces before and after contact with walls %%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Moments  Mx = Frecov1(4)

axes(ha(5)); plot([0 2*pi],[0 0]); %  need this to get numbers on x axis
hold on; 
%for i=1:j-1
for k=seg1:seg2
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),Frecov1(xtickmin(k):xtickmax(k),4),'r-','LineWidth',1); 
  %str = sprintf('k = %d, xtickmin= %d   xtickmax=%d\n',k,xtickmin(k),xtickmax(k));
  %display(str);
end
ylabel('M (mN-M)','FontSize', 14, 'FontName', 'CMU Serif');
axis([0,2*pi,-50,50]); 

axes(ha(6));  plot([0 2*pi],[0 0]); % need this to get numbers on x axis
hold on; 
for k=seg3:seg4
  plot(mod(rightLegPos(xtickmin(k):xtickmax(k)),2*pi),Frecov1(xtickmin(k):xtickmax(k),4),'r-','LineWidth',1); 
  %str = sprintf('k = %d, xtickmin= %d   xtickmax=%d\n',k,xtickmin(k),xtickmax(k));
  %display(str);
end
axis([0,2*pi,-50,50])
set(ha(6),'YTickLabel','') % only label on left plot
legend('M_x')



%%%%%%%%%%
%%%% change fonts, line widths for all plots, etc:
set(ha(1:4),'XTickLabel','') % only 1 time label
for i = 1:6
    axes(ha(i));
    set(gca,'FontName','CMU Serif','FontSize',14);
    %axis([0,10,-1.5,1.5]);
    hold on
    temp = get(gca,'XTick');
    plot([temp(1),temp(end)],[0,0],'k','LineWidth',1);
    grid on
    
end
axes(ha(5));
xlabel('Rt Leg position (rad)','FontSize', 18, 'FontName', 'CMU Serif');
axes(ha(6));
xlabel('Rt Leg position (rad)','FontSize', 18, 'FontName', 'CMU Serif');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate discrete fourier series
k=1; % frequency index
l=seg1; 
x1=Frecov1(xtickmin(l):xtickmax(l+3),2);
N=xtickmax(l+3)-xtickmin(l);  % number of samples
omega = (0:N)*2*pi/N;

% FS for first segment
for k=1:17  % frequency components
    total = 0;
    for i=1:N-1
        total = total + x1(i)*exp(1j*(k-1)*omega(i));
    end
    a(k) = total/(N);
end

l=seg3; 
x2=Frecov1(xtickmin(l):xtickmax(l+3),2); % Fy
N=xtickmax(l+3)-xtickmin(l);  % number of samples
omega = (0:N)*2*pi/N;
%FS for second segment
for k=1:17  % frequency components
    total = 0;
    for i=1:N-1
        total = total + x2(i)*exp(1j*(k-1)*omega(i));
    end
    b(k) = total/(N);
end

% print magnitude of coefficients
figure(5)
clf;
ha = tight_subplot(2,1,[.05 0.05],[.1 .1],[.1 .1]);
% ha = tight_subplot(rows, cols, [gapy gapx], marg_h, marg_w)
axes(ha(1));
stem(0:16,abs(a),'k','linewidth',2)
ylabel('{|a_k|}','FontSize', 14, 'FontName', 'CMU Serif');

axes(ha(2));
stem(0:16,abs(b),'k','linewidth',2)
ylabel('{|b_k|}','FontSize', 14, 'FontName', 'CMU Serif');
xlabel('frequency k','FontSize', 18, 'FontName', 'CMU Serif');

for i=1:2
    axes(ha(i))
    axis([-0.5,16.5,0,0.3])
    set(gca,'FontName','CMU Serif','FontSize',14);
end
