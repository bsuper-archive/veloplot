% maketerrainplot
% pulled from RobotGrassShellForce.m to shorten file

if processflag==1    
    
else if processflag==2
        
        h101=figure(101);clf;set(gcf,'color','w');
        
        subplot(1,2,1);hold all;box on;
        errorbar(layerind,avgFFx1mean,avgFFx1std,'b-','linewidth',lnwz);
        errorbar(layerind,avgFFy1mean,avgFFy1std,'r-','linewidth',lnwz);
        errorbar(layerind,avgFFz1mean,avgFFz1std,'g-','linewidth',lnwz);
        if plottrials==1
            for ii=1:lff/3
                plot(layerind(ii),avgFFx1((ii-1)*3+1),'bo','markersize',mksz);
                plot(layerind(ii),avgFFx1((ii-1)*3+2),'bs','markersize',mksz);
                plot(layerind(ii),avgFFx1((ii-1)*3+3),'b^','markersize',mksz);
                plot(layerind(ii),avgFFy1((ii-1)*3+1),'ro','markersize',mksz);
                plot(layerind(ii),avgFFy1((ii-1)*3+2),'rs','markersize',mksz);
                plot(layerind(ii),avgFFy1((ii-1)*3+3),'r^','markersize',mksz);
                plot(layerind(ii),avgFFz1((ii-1)*3+1),'go','markersize',mksz);
                plot(layerind(ii),avgFFz1((ii-1)*3+2),'gs','markersize',mksz);
                plot(layerind(ii),avgFFz1((ii-1)*3+3),'g^','markersize',mksz);
                
%                 plot(layerind(ii),avgFFx2((ii-1)*3+1),'bo','markersize',mksz,'markerfacecolor','b');
%                 plot(layerind(ii),avgFFx2((ii-1)*3+2),'bs','markersize',mksz,'markerfacecolor','b');
%                 plot(layerind(ii),avgFFx2((ii-1)*3+3),'b^','markersize',mksz,'markerfacecolor','b');
%                 plot(layerind(ii),avgFFy2((ii-1)*3+1),'ro','markersize',mksz,'markerfacecolor','r');
%                 plot(layerind(ii),avgFFy2((ii-1)*3+2),'rs','markersize',mksz,'markerfacecolor','r');
%                 plot(layerind(ii),avgFFy2((ii-1)*3+3),'r^','markersize',mksz,'markerfacecolor','r');
%                 plot(layerind(ii),avgFFz2((ii-1)*3+1),'go','markersize',mksz,'markerfacecolor','g');
%                 plot(layerind(ii),avgFFz2((ii-1)*3+2),'gs','markersize',mksz,'markerfacecolor','g');
%                 plot(layerind(ii),avgFFz2((ii-1)*3+3),'g^','markersize',mksz,'markerfacecolor','g');
            end
        end
        line([0 5],[0 0],'linestyle',':','color','k');        
        xlim([-0.2 5.2]);
        ylim([-0.08 0.16]);
        ylabel('Force (N)','fontsize',ftsz2);
        xlabel('Layer','fontsize',ftsz2);
        legend('Forward','Left','Upward','location','northwest');
        set(gca,'fontsize',ftsz2);
        
        subplot(1,2,2);hold all;box on;
        errorbar(layerind,avgNNx1mean,avgNNx1std,'b-','linewidth',lnwz);
        errorbar(layerind,avgNNy1mean,avgNNy1std,'r-','linewidth',lnwz);
        errorbar(layerind,avgNNz1mean,avgNNz1std,'g-','linewidth',lnwz);
        if plottrials==1
            for ii=1:lff/3
                plot(layerind(ii),avgNNx1((ii-1)*3+1),'bo','markersize',mksz);
                plot(layerind(ii),avgNNx1((ii-1)*3+2),'bs','markersize',mksz);
                plot(layerind(ii),avgNNx1((ii-1)*3+3),'b^','markersize',mksz);
                plot(layerind(ii),avgNNy1((ii-1)*3+1),'ro','markersize',mksz);
                plot(layerind(ii),avgNNy1((ii-1)*3+2),'rs','markersize',mksz);
                plot(layerind(ii),avgNNy1((ii-1)*3+3),'r^','markersize',mksz);
                plot(layerind(ii),avgNNz1((ii-1)*3+1),'go','markersize',mksz);
                plot(layerind(ii),avgNNz1((ii-1)*3+2),'gs','markersize',mksz);
                plot(layerind(ii),avgNNz1((ii-1)*3+3),'g^','markersize',mksz);
            end
        end
        line([0 5],[0 0],'linestyle',':','color','k');
        xlim([-0.2 5.2]);
        ylabel('Torque (mN*m)','fontsize',ftsz2);
        xlabel('Layer','fontsize',ftsz2);
        legend('Roll right','Pitch down','Yaw left','location','southwest');
        set(gca,'fontsize',ftsz2);
        
        saveas(h101,strcat(dirname1,'vary layer'),'fig');
        saveas(h101,strcat(dirname1,'vary layer'),'jpg');
        
    else if processflag==3
            
            h101=figure(101);clf;set(gcf,'color','w');
        
            subplot(1,2,1);hold all;box on;
            errorbar(spacingind,avgFFx1mean,avgFFx1std,'b-','linewidth',lnwz);
            errorbar(spacingind,avgFFy1mean,avgFFy1std,'r-','linewidth',lnwz);
            errorbar(spacingind,avgFFz1mean,avgFFz1std,'g-','linewidth',lnwz);
            if plottrials==1
                for ii=1:lff/3
                    plot(spacingind(ii),avgFFx1((ii-1)*3+1),'bo','markersize',mksz);
                    plot(spacingind(ii),avgFFx1((ii-1)*3+2),'bs','markersize',mksz);
                    plot(spacingind(ii),avgFFx1((ii-1)*3+3),'b^','markersize',mksz);
                    plot(spacingind(ii),avgFFy1((ii-1)*3+1),'ro','markersize',mksz);
                    plot(spacingind(ii),avgFFy1((ii-1)*3+2),'rs','markersize',mksz);
                    plot(spacingind(ii),avgFFy1((ii-1)*3+3),'r^','markersize',mksz);
                    plot(spacingind(ii),avgFFz1((ii-1)*3+1),'go','markersize',mksz);
                    plot(spacingind(ii),avgFFz1((ii-1)*3+2),'gs','markersize',mksz);
                    plot(spacingind(ii),avgFFz1((ii-1)*3+3),'g^','markersize',mksz);
                end
            end
            line([9 12],[0 0],'linestyle',':','color','k');        
            xlim([8.8 12.2]);
%             ylim([-0.08 0.16]);
            ylabel('Force (N)','fontsize',ftsz2);
            xlabel('Spacing (cm)','fontsize',ftsz2);
            legend('Forward','Left','Upward','location','northeast');
            set(gca,'fontsize',ftsz2);

            subplot(1,2,2);hold all;box on;
            errorbar(spacingind,avgNNx1mean,avgNNx1std,'b-','linewidth',lnwz);
            errorbar(spacingind,avgNNy1mean,avgNNy1std,'r-','linewidth',lnwz);
            errorbar(spacingind,avgNNz1mean,avgNNz1std,'g-','linewidth',lnwz);
            if plottrials==1
                for ii=1:lff/3
                    plot(spacingind(ii),avgNNx1((ii-1)*3+1),'bo','markersize',mksz);
                    plot(spacingind(ii),avgNNx1((ii-1)*3+2),'bs','markersize',mksz);
                    plot(spacingind(ii),avgNNx1((ii-1)*3+3),'b^','markersize',mksz);
                    plot(spacingind(ii),avgNNy1((ii-1)*3+1),'ro','markersize',mksz);
                    plot(spacingind(ii),avgNNy1((ii-1)*3+2),'rs','markersize',mksz);
                    plot(spacingind(ii),avgNNy1((ii-1)*3+3),'r^','markersize',mksz);
                    plot(spacingind(ii),avgNNz1((ii-1)*3+1),'go','markersize',mksz);
                    plot(spacingind(ii),avgNNz1((ii-1)*3+2),'gs','markersize',mksz);
                    plot(spacingind(ii),avgNNz1((ii-1)*3+3),'g^','markersize',mksz);
                end
            end
            line([9 12],[0 0],'linestyle',':','color','k');
            xlim([8.8 12.2]);
            ylim([-10 11]);
            ylabel('Torque (mN*m)','fontsize',ftsz2);
            xlabel('Spacing (cm)','fontsize',ftsz2);
            legend('Roll right','Pitch down','Yaw left','location','northeast');
            set(gca,'fontsize',ftsz2);

            saveas(h101,strcat(dirname1,'vary spacing'),'fig');
            saveas(h101,strcat(dirname1,'vary spacing'),'jpg');            
            
        else            
            
            h101=figure(101);clf;set(gcf,'color','w');
        
            subplot(1,2,1);hold all;box on;
            errorbar(freqind,avgFFx1mean(4:6),avgFFx1std(4:6),'b-','linewidth',lnwz2);
            errorbar(freqind,avgFFy1mean(4:6),avgFFy1std(4:6),'r-','linewidth',lnwz2);
            errorbar(freqind,avgFFz1mean(4:6),avgFFz1std(4:6),'g-','linewidth',lnwz2);
            errorbar(freqind,avgFFx1mean(1:3),avgFFx1std(1:3),'b--','linewidth',lnwz);
            errorbar(freqind,avgFFy1mean(1:3),avgFFy1std(1:3),'r--','linewidth',lnwz);
            errorbar(freqind,avgFFz1mean(1:3),avgFFz1std(1:3),'g--','linewidth',lnwz);            
            if plottrials==1
                for ii=1:lff/6
                    plot(freqind(ii),avgFFx1(9+(ii-1)*3+1),'bo','markersize',mksz2);
                    plot(freqind(ii),avgFFx1(9+(ii-1)*3+2),'bs','markersize',mksz2);
                    plot(freqind(ii),avgFFx1(9+(ii-1)*3+3),'b^','markersize',mksz2);
                    plot(freqind(ii),avgFFy1(9+(ii-1)*3+1),'ro','markersize',mksz2);
                    plot(freqind(ii),avgFFy1(9+(ii-1)*3+2),'rs','markersize',mksz2);
                    plot(freqind(ii),avgFFy1(9+(ii-1)*3+3),'r^','markersize',mksz2);
                    plot(freqind(ii),avgFFz1(9+(ii-1)*3+1),'go','markersize',mksz2);
                    plot(freqind(ii),avgFFz1(9+(ii-1)*3+2),'gs','markersize',mksz2);
                    plot(freqind(ii),avgFFz1(9+(ii-1)*3+3),'g^','markersize',mksz2);
                    
                    plot(freqind(ii),avgFFx1((ii-1)*3+1),'bo','markersize',mksz);
                    plot(freqind(ii),avgFFx1((ii-1)*3+2),'bs','markersize',mksz);
                    plot(freqind(ii),avgFFx1((ii-1)*3+3),'b^','markersize',mksz);
                    plot(freqind(ii),avgFFy1((ii-1)*3+1),'ro','markersize',mksz);
                    plot(freqind(ii),avgFFy1((ii-1)*3+2),'rs','markersize',mksz);
                    plot(freqind(ii),avgFFy1((ii-1)*3+3),'r^','markersize',mksz);
                    plot(freqind(ii),avgFFz1((ii-1)*3+1),'go','markersize',mksz);
                    plot(freqind(ii),avgFFz1((ii-1)*3+2),'gs','markersize',mksz);
                    plot(freqind(ii),avgFFz1((ii-1)*3+3),'g^','markersize',mksz);
                end
            end
            line([7 13],[0 0],'linestyle',':','color','k');        
            xlim([6.8 13.2]);
            ylim([-0.23 0.2]);
            ylabel('Force (N)','fontsize',ftsz2);
            xlabel('Frequency (Hz)','fontsize',ftsz2);
            legend('Forward','Left','Upward','location','southwest');
            set(gca,'fontsize',ftsz2);

            subplot(1,2,2);hold all;box on;
            errorbar(freqind,avgNNx1mean(4:6),avgNNx1std(4:6),'b-','linewidth',lnwz2);
            errorbar(freqind,avgNNy1mean(4:6),avgNNy1std(4:6),'r-','linewidth',lnwz2);
            errorbar(freqind,avgNNz1mean(4:6),avgNNz1std(4:6),'g-','linewidth',lnwz2);
            errorbar(freqind,avgNNx1mean(1:3),avgNNx1std(1:3),'b--','linewidth',lnwz);
            errorbar(freqind,avgNNy1mean(1:3),avgNNy1std(1:3),'r--','linewidth',lnwz);
            errorbar(freqind,avgNNz1mean(1:3),avgNNz1std(1:3),'g--','linewidth',lnwz);
            if plottrials==1
                for ii=1:lff/6
                    plot(freqind(ii),avgNNx1(9+(ii-1)*3+1),'bo','markersize',mksz2);
                    plot(freqind(ii),avgNNx1(9+(ii-1)*3+2),'bs','markersize',mksz2);
                    plot(freqind(ii),avgNNx1(9+(ii-1)*3+3),'b^','markersize',mksz2);
                    plot(freqind(ii),avgNNy1(9+(ii-1)*3+1),'ro','markersize',mksz2);
                    plot(freqind(ii),avgNNy1(9+(ii-1)*3+2),'rs','markersize',mksz2);
                    plot(freqind(ii),avgNNy1(9+(ii-1)*3+3),'r^','markersize',mksz2);
                    plot(freqind(ii),avgNNz1(9+(ii-1)*3+1),'go','markersize',mksz2);
                    plot(freqind(ii),avgNNz1(9+(ii-1)*3+2),'gs','markersize',mksz2);
                    plot(freqind(ii),avgNNz1(9+(ii-1)*3+3),'g^','markersize',mksz2);
                    
                    plot(freqind(ii),avgNNx1((ii-1)*3+1),'bo','markersize',mksz);
                    plot(freqind(ii),avgNNx1((ii-1)*3+2),'bs','markersize',mksz);
                    plot(freqind(ii),avgNNx1((ii-1)*3+3),'b^','markersize',mksz);
                    plot(freqind(ii),avgNNy1((ii-1)*3+1),'ro','markersize',mksz);
                    plot(freqind(ii),avgNNy1((ii-1)*3+2),'rs','markersize',mksz);
                    plot(freqind(ii),avgNNy1((ii-1)*3+3),'r^','markersize',mksz);
                    plot(freqind(ii),avgNNz1((ii-1)*3+1),'go','markersize',mksz);
                    plot(freqind(ii),avgNNz1((ii-1)*3+2),'gs','markersize',mksz);
                    plot(freqind(ii),avgNNz1((ii-1)*3+3),'g^','markersize',mksz);
                end
            end
            line([7 13],[0 0],'linestyle',':','color','k');
            xlim([6.8 13.2]);
            ylim([-10 18]);
            ylabel('Torque (mN*m)','fontsize',ftsz2);
            xlabel('Frequency (Hz)','fontsize',ftsz2);
            legend('Roll right','Pitch down','Yaw left','location','northwest');
            set(gca,'fontsize',ftsz2);

            saveas(h101,strcat(dirname1,'vary stiffness & frequency'),'fig');
            saveas(h101,strcat(dirname1,'vary stiffness & frequency'),'jpg');
            
        end
    end
end
