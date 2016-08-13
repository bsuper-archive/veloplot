%%
% makemovie from RobotGrassShellForce.m
% file split to make more manageable
cd(pathmov);  % in videos/plots directory
        for ii=1:movstep:indend(jj)

            if ii-avgwidth*samplingfreq/2<1
                avgind=1:(1+avgwidth*samplingfreq);
            else if ii+avgwidth*samplingfreq/2>length(timez)
                    avgind=(length(timez)-avgwidth*samplingfreq):length(timez);
                else
                    avgind=(ii-avgwidth*samplingfreq/2):(ii+avgwidth*samplingfreq/2);
                end
            end
                        
            h3=figure(3);clf;hold all;
            set(gcf,'color','w');
            
            subplot(2,1,1);hold all;box on;
            plot(timez,FFx,'b-','linewidth',lnwz);
            plot(timez,FFy,'r-','linewidth',lnwz);
            plot(timez,FFz,'g-','linewidth',lnwz);
            plot(timez(ii),FFx(ii),'bo','markerfacecolor','b');
            plot(timez(ii),FFy(ii),'ro','markerfacecolor','r');
            plot(timez(ii),FFz(ii),'go','markerfacecolor','g');
            line([start1(jj) end1(jj)]/30,[avgFFx1(jj) avgFFx1(jj)],'color','b','linewidth',lnwz,'linestyle',':');
            line([start1(jj) end1(jj)]/30,[avgFFy1(jj) avgFFy1(jj)],'color','r','linewidth',lnwz,'linestyle',':');
            line([start1(jj) end1(jj)]/30,[avgFFz1(jj) avgFFz1(jj)],'color','g','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgFFx2(jj) avgFFx2(jj)],'color','b','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgFFy2(jj) avgFFy2(jj)],'color','r','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgFFz2(jj) avgFFz2(jj)],'color','g','linewidth',lnwz,'linestyle',':');
            line([start1(jj) start1(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
            line([end1(jj) end1(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
            line([start2(jj) start2(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
            line([end2(jj) end2(jj)]/30,[min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])],'linestyle',':','color','k');
            line([0 runlength],[0 0],'linestyle',':','color','k');
            xlim([0 runlength*1.35]);
            ylim([min([min(FFx) min(FFy) min(FFz)]) max([max(FFx) max(FFy) max(FFz)])]);
            ylabel('Force (N)','fontsize',ftsz);
            xlabel('Time (s)','fontsize',ftsz);
            legend('Forward','Left','Upward','location','southeast');
            set(gca,'fontsize',ftsz);
            title(nameinnoext,'Interpreter','none');

            subplot(2,1,2);hold all;box on;                                
            plot(timez,NNx,'b-','linewidth',lnwz);
            plot(timez,NNy,'r-','linewidth',lnwz);
            plot(timez,NNz,'g-','linewidth',lnwz);
            plot(timez(ii),NNx(ii),'bo','markerfacecolor','b');
            plot(timez(ii),NNy(ii),'ro','markerfacecolor','r');
            plot(timez(ii),NNz(ii),'go','markerfacecolor','g');
            line([start1(jj) end1(jj)]/30,[avgNNx1(jj) avgNNx1(jj)],'color','b','linewidth',lnwz,'linestyle',':');
            line([start1(jj) end1(jj)]/30,[avgNNy1(jj) avgNNy1(jj)],'color','r','linewidth',lnwz,'linestyle',':');
            line([start1(jj) end1(jj)]/30,[avgNNz1(jj) avgNNz1(jj)],'color','g','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgNNx2(jj) avgNNx2(jj)],'color','b','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgNNy2(jj) avgNNy2(jj)],'color','r','linewidth',lnwz,'linestyle',':');
            line([start2(jj) end2(jj)]/30,[avgNNz2(jj) avgNNz2(jj)],'color','g','linewidth',lnwz,'linestyle',':');
            line([start1(jj) start1(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
            line([end1(jj) end1(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
            line([start2(jj) start2(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
            line([end2(jj) end2(jj)]/30,[min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])],'linestyle',':','color','k');
            line([0 runlength],[0 0],'linestyle',':','color','k');
            xlim([0 runlength*1.35]);
            ylim([min([min(NNx) min(NNy) min(NNz)]) max([max(NNx) max(NNy) max(NNz)])]);
            ylabel('Torque (mN*m)','fontsize',ftsz);
            xlabel('Time (s)','fontsize',ftsz);
            legend('Roll right','Pitch down','Yaw left','location','southeast');
            set(gca,'fontsize',ftsz);
 
 %return % for debugging
            
            set(gcf,'units','pixels');
            set(gcf,'Position',[100,100,800,380]);
            
            mkdir(nameinnoext);
            %cd(nameinnoext);
            % use filename without all the cd stuff'
            filejpg = strcat(nameinnoext,'/',strcat(num2str(ii,'%04d'),'.jpg'))
            %export_fig(strcat(num2str(ii,'%04d'),'.jpg'),gcf,'-nocrop');
            export_fig(filejpg,gcf,'-nocrop');
                       
        end
    end
end
