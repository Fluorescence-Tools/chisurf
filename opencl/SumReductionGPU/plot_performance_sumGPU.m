%% Plot performances for GPU Sum Reduction %%

x = load('performances.txt');

% Get Runtimes
for i = 1:6
time_seq(1:5,i) = x((i-1)*5+1:i*5,3);
time_gpu(1:5,i) = x((i-1)*5+1:i*5,4);
speedup(1:5,i) = time_seq(1:5,i)./time_gpu(1:5,i);
end 

% X axis
sizeArray = [1024 10240 102400 1024000 10240000 102400000]
figure(1);

% Get Histogram
h = bar(log10(sizeArray),log10(speedup(1:5,:)')); % get histogram

% Log10 for x-axis and xtick
set(gca,'Xtick',log10(1024):1:log10(1.024*10^8))
set(gca,'Xticklabel',10.^get(gca,'Xtick'));
set(h(1),'facecolor',[0.5 0.5 1]);
set(h(2),'facecolor',[1 0.5 0.5]);
set(h(3),'facecolor',[0.5 1 0.5]); 
set(h(4),'facecolor',[0.5 0.5 0.5]); 
set(h(5),'facecolor',[1 0.5 1]); 
hPatch = findobj(h,'Type','patch');
set(hPatch,'facealpha',1); 
grid on;
title('Benchmark GPU vs CPU');

% Size of WorkGroup
h = legend('N=16','N=32','N=64','N=128','N=256');
v = get(h,'title');
set(v,'string','WorkGroup size');

% Place legend
rect = [0.6,0.25,0.2,0.2];
set(h,'Position',rect,'color','w');
hPatch = findobj(h,'Type','patch');
set(hPatch,'facealpha',1); 
xlabel('log(Array size)');
ylabel('log(Speedup)');

% Make right y-axis visible
ax1 = gca;
ax2 = axes('Position', get(ax1, 'Position'));
set(ax2, 'YAxisLocation', 'right', 'Color', 'none', 'XTickLabel', []);
set(ax2, 'YLim', get(ax1, 'YLim'));
set(ax2,'XTick',[]);
