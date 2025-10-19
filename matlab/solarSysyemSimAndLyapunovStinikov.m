clear;
close all;
clc;

% ====== Unidades astronómicas ======
T = 1/365;               % 1 día
G = 4*pi^2;

% ====== Sitnikov minimal (2D: Y ~ eje Z) ======
N = 3;

M = [0.5, 0.5, 0];      % dos primarios iguales y partícula test (masa 0)

a = 0.5;                % radio de la órbita de cada primario [UA] (separación = 1 UA)

% Primarios en ±a sobre X, con velocidades circulares opuestas sobre Y
omega = 2*pi*sqrt( (M(1)+M(2)) / ( (2*a)^3 ) ) * (2^(3/2)); 
% Nota: En unidades keplerianas, puedes usar directamente:
% omega = 2*pi*sqrt( (M(1)+M(2)) / ( (2*a)^3 ) );  % periodo de la binaria de separación 2a

R = [ -a, 0;            % primario 1
       a, 0;            % primario 2
       0, 1e-3];        % partícula en "z"~Y, pequeña separación inicial

Rf = R;

v1 = [ 0, -omega*a ];   % primario 1
v2 = [ 0,  omega*a ];   % primario 2
vz0= 0.0;               % velocidad inicial de la partícula en Y (ajusta 0–1e-3)

V = [v1; v2; 0, vz0];
V2= V;

% ====== Lyapunov ======
steps = 20000;       % deja correr más pasos (rápido)
m     = 100;          % renorm cada 15 días
tau   = m*T;
delta0= 1e-6;        % [UA]
alpha = 0;           % solo posiciones
sigma = 0; count_renorm = 0;

% Perturbación: pequeña en Y sobre la partícula
dR = zeros(size(R)); dV = zeros(size(V));
dR(3,:) = [0, delta0];
d = sqrt(sum(dR(:).^2) + (alpha^2)*sum(dV(:).^2));
R2 = R + (delta0/d)*dR; V2 = V + (delta0/d)*dV;

% Gráfica en UA
% Configuracion plot
% ==== CONFIGURACION DE SUBPLOTS ====
lims = [-1.5, 1.5, -1.5, 1.5];
names  = {'Primary 1','Primary 2','Test mass'};
colors = lines(N);

% Figura con 3 paneles: base, perturbado, overlay
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% Panel 1: Sistema base
ax1 = nexttile(1); axis(ax1,'equal'); axis(ax1,lims); grid(ax1,'on'); hold(ax1,'on');
title(ax1,'Base');
htraj_base = gobjects(N,1); hdot_base = gobjects(N,1);
for i=1:N
    htraj_base(i) = animatedline(ax1,'LineWidth',1.0,'Color',colors(i,:));
    hdot_base(i)  = plot(ax1,R(i,1),R(i,2),'o','MarkerSize',4, ...
                         'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none');
end
legend(ax1,hdot_base,names,'Location','southoutside','Orientation','horizontal');

% Panel 2: Sistema perturbado
ax2 = nexttile(2); axis(ax2,'equal'); axis(ax2,lims); grid(ax2,'on'); hold(ax2,'on');
title(ax2,'Perturbado');
htraj_pert = gobjects(N,1); hdot_pert = gobjects(N,1);
for i=1:N
    htraj_pert(i) = animatedline(ax2,'LineWidth',1.0,'Color',colors(i,:));
    hdot_pert(i)  = plot(ax2,R2(i,1),R2(i,2),'o','MarkerSize',4, ...
                         'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none');
end
legend(ax2,hdot_pert,names,'Location','southoutside','Orientation','horizontal');

% Panel 3: Superpuesto (base vs perturbado)
ax3 = nexttile(3); axis(ax3,'equal'); axis(ax3,lims); grid(ax3,'on'); hold(ax3,'on');
title(ax3,'Superpuesto');

% Paleta para base y otra distinta para perturbado
colors_base = lines(N);
colors_pert = turbo(N);           
htraj_ov_base = gobjects(N,1); htraj_ov_pert = gobjects(N,1);
hdot_ov_base  = gobjects(N,1); hdot_ov_pert  = gobjects(N,1);
for i=1:N
    % Trayectorias
    htraj_ov_base(i) = animatedline(ax3,'LineWidth',1.2,'Color',colors_base(i,:), ...
        'DisplayName', sprintf('%s Base', names{i}));
    htraj_ov_pert(i) = animatedline(ax3,'LineWidth',1.2,'LineStyle','--','Color',colors_pert(i,:), ...
        'DisplayName', sprintf('%s Pert.', names{i}));
    % Marcadores de posición actual
    hdot_ov_base(i) = plot(ax3, R(i,1), R(i,2), 'o', 'MarkerSize',4, ...
        'MarkerFaceColor',colors_base(i,:), 'MarkerEdgeColor','none', ...
        'DisplayName', sprintf('%s Base', names{i}));
    hdot_ov_pert(i) = plot(ax3, R2(i,1), R2(i,2), '^', 'MarkerSize',4, ...
        'MarkerFaceColor',colors_pert(i,:), 'MarkerEdgeColor','none', ...
        'DisplayName', sprintf('%s Pert.', names{i}));
end
% Legend automática con todos los DisplayName
lg = legend(ax3,'show','Location','eastoutside','Orientation','horizontal');
lg.NumColumns = 2;   % opcional: 2 columnas (Base / Pert.)

% Segunda Figura
figDist = figure('Name','Crecimiento de la perturbación','NumberTitle','off');
axDist  = axes(figDist); hold(axDist,'on'); grid(axDist,'on');
title(axDist,'||R_2 - R|| vs tiempo');
xlabel(axDist,'Paso (k)'); ylabel(axDist,'Distancia [m]');
set(axDist,'YScale','log'); % semilog-Y
hline_dist = animatedline(axDist,'LineWidth',1.5);

for k=1:steps
    %Sistema Sin perturbaciones
    for i = 1:N
        A = [0 0];
        for j = 1 : N
            if i~= j
                r = R(i,:)-R(j,:);
                r2 = norm(r);
                ru = r/r2;
                A = A - G * M(j) * ru /(r2^2);
            end
        end
        V(i,:) = V(i,:) + A * T;
        Rf(i,:) = R(i,:) + V(i,:) * T;
    end
    R = Rf;
    
    % ==== DIBUJO EN LOS 3 SUBPLOTS ====
    for i=1:N
        % Base
        addpoints(htraj_base(i), R(i,1), R(i,2));
        set(hdot_base(i), 'XData', R(i,1), 'YData', R(i,2));
        % Perturbado
        addpoints(htraj_pert(i), R2(i,1), R2(i,2));
        set(hdot_pert(i), 'XData', R2(i,1), 'YData', R2(i,2));
        % Overlay
        addpoints(htraj_ov_base(i), R(i,1),  R(i,2));
        addpoints(htraj_ov_pert(i), R2(i,1), R2(i,2));
        set(hdot_ov_base(i), 'XData', R(i,1),  'YData', R(i,2));
        set(hdot_ov_pert(i), 'XData', R2(i,1), 'YData', R2(i,2));

    end
    
    if mod(k,5)==0
        drawnow limitrate
        % pause(0.02)  % <- si lo quieres más lento
    end
    
    % Sistema perturbado
    for i=1:N
        A = [0 0];
        for j = 1:N
            if i ~= j
                r = R2(i,:) - R2(j,:);
                r2 = norm(r);
                ru = r / r2;
                A = A - G * M(j) * ru /(r2^2);
            end
        end
        V2(i,:) = V2(i,:) + A * T;
        R2(i,:) = R2(i,:) + V2(i,:) * T;
    end

    % Distancia entre trayectorias (solo posiciones)
    diffR = R2 - R;
    dist  = sqrt(sum(diffR(:).^2)); % norma euclidiana de todas las posiciones
    addpoints(hline_dist, k, dist);

    yline(axDist, delta0, '--', '\delta_0');
    % Curva normalizada (sin cambiar la principal)
    hline_norm = animatedline(axDist,'LineStyle',':','LineWidth',1.2);
    addpoints(hline_norm, k, dist / delta0);


    %Renormalizacion y acumulado
    if mod(k,m)==0
    %if (d > grow_up*delta0) || (d < grow_down*delta0) || mod(k, m)==0
        dR = R2 - R;
        dV = V2 -V;
        d = sqrt(sum(sum(dR.^2)) + (alpha^2)*sum(sum(dV.^2)));
        sigma = sigma + log(d/delta0);
        scale = delta0 / d;
        R2 = R + scale*dR;
        V2 = V + scale*dV;
        count_renorm = count_renorm + 1;
    end

    %Plot
    %if mod(k,5)==0
    %    plot(R(1:N,1),R(1:N,2),'ob'); axis([-1E12,1E12,-1E12,1E12]);
    %    title(sprintf('Paso %d  |  renorm=%d',k,count_renorm)); pause(0.01);
    %end
end

lambda = sigma / (count_renorm*tau);                 % [1/s]
lambda_per_year = lambda*(365.25*86400);  % [1/año]
fprintf('Lyapunov ~ %.3e 1/s   (%.3e 1/año)\n',lambda,lambda_per_year);