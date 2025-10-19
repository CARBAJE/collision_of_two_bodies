clear;
close all;
clc
%hold on

% COnstantes y parametros
T=1*86400;%incremento en tiempo 1 dia=86400 S;
G=6.67E-11;%m3/s2Kg gravedad

% Elementos del sistema de n cuerpos ; n = 6
N=6;%numero de part?culas: sol mercurio venus tierra marte jupiter
M=[333000,.0558,.815,1,1.07,310]*5.98E24;%masas S M V T M J
R=[0 0;57.9 0;108 0;150 0;228 0;778 0]*1E9;%posici?n inicial
Rf=R;%auxiliar
V=[0 0;0 47.9;0 35;0 29.8;0 24.1;0 13.1]*1E3;%velocidad inicial (tangente)

while(1);%calcula la siguiente iteraci?n
    for i=1:N;%para cada part?culas
        A=[0 0];%inicializa aceleraci?n
        for j=1:N%para la fuerza con todas las particulas
            if i~=j% menos ella misma
                r=R(i,:)-R(j,:);%vector de la part. i a la j
                r2=norm(r);%magnitud del vector r
                ru=r/r2;%calcula unitario
                A=A-G*M(j)*ru/(r2*r2);%suma de aceleraciones
            end
        end
        V(i,:)=V(i,:)+A*T;%%%%%%%%Sol. de la Ec. Diferencial
        Rf(i,:)=R(i,:)+V(i,:)*T;%%
    end
    R=Rf;%actualiza posiciones iniciales
    plot(R(1:N,1),R(1:N,2),'ob')%grafica nuevas posiciones
    axis([-1E12,1E12,-1E12,1E12])%limites de la gr?fica
    pause(0.01);%permite visualizar antes de continuar el c?lculo
end