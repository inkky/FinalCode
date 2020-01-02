# -*- coding: utf-8 -*-
# @Time    : 2018/6/6 17:01
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
error vs tpaa,paa
'''
import numpy as np
import matplotlib.pyplot as plt


paa = np.loadtxt('PAAresult/error_paa.txt', delimiter='\n')
bt = np.loadtxt('PAAresult/error_btpaa.txt', delimiter='\n')
saxtd = np.loadtxt('PAAresult/error_saxtd.txt', delimiter='\n')
cosin = np.loadtxt('PAAresult/error_cos.txt', delimiter='\n')
papr = np.loadtxt('PAAresult/error_papr.txt', delimiter='\n')
rdos = np.loadtxt('PAAresult/error_rdos.txt', delimiter='\n')
esax = np.loadtxt('PAAresult/error_esax.txt', delimiter='\n')
dlde = np.loadtxt('PAAresult/error_dlde.txt', delimiter='\n')





x=np.linspace(0,0.5)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(6, 3)
# plt.scatter(eu, nt)
# for i in range(len(papr)):
#     if rdos[i] > bt[i]:
#         plt.scatter(rdos[i], bt[i],marker='o',c='r')
#     if rdos[i] < bt[i]:
#         plt.scatter(rdos[i],bt[i],marker='^',c='b')
#     if rdos[i] == bt[i]:
#         plt.scatter(rdos[i], bt[i], marker='s',c='g')

for i in range(len(papr)):
    if esax[i] > bt[i]:
        plt.scatter(dlde[i], bt[i],marker='o',c='r')
    if esax[i] < bt[i]:
        plt.scatter(dlde[i],bt[i],marker='^',c='b')
    if esax[i] == bt[i]:
        plt.scatter(dlde[i], bt[i], marker='s',c='g')

# for i in range(len(papr)):
#     if papr[i] > bt[i]:
#         plt.scatter(papr[i], bt[i],marker='o',c='r')
#     if papr[i] < bt[i]:
#         plt.scatter(papr[i],bt[i],marker='^',c='b')
#     if papr[i] == bt[i]:
#         plt.scatter(papr[i], bt[i], marker='s',c='g')
plt.xlabel('Error rate of DLDE')
plt.ylabel('Error rate of TPAA')
plt.plot(x,x,'--')
plt.xlim(0,0.5)
plt.ylim(0,0.5)
plt.tight_layout()
plt.text(0.05,0.3,'$DLDE\ distance\ region$',fontdict={'color': 'b'})
plt.text(0.3,0.1,'$TPAA\ distance\ region$',fontdict={'color': 'r'})
plt.savefig('PAAimg/dlde_vs_bt.png', dpi=300)
plt.show()
