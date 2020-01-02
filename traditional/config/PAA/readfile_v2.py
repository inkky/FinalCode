# -*- coding: utf-8 -*-
# @Time    : 2018/5/10 9:24
# @Author  : Inkky
# @Email   : yingyang_chen@163.com
'''
change all old ucr data sets into .txt

'''
import os

filedir = os.getcwd()+'/UCR_TS_Archive_2015'

file=os.listdir(filedir)
# print('files',file)
# f=open('/UCR_TS_Archive_2015/test.txt','w')

# 将data转成.txt文件
# for dirname in file:
#     if not dirname.startswith('.') or dirname.endswith('.txt'):
#         # print(dirname)
#         f=os.listdir(os.getcwd()+'/UCR_TS_Archive_2015/'+dirname)
#         for filename in f:
#             oldname=os.getcwd()+'/UCR_TS_Archive_2015/'+dirname+'/'+filename
#             newname=os.getcwd()+'/UCR_TS_Archive_2015/'+dirname+'/'+filename+'.txt'
#             # print(filename)
#             # print(oldname,newname)
#             os.rename(oldname,newname)


f=os.getcwd()+'/UCR_TS_Archive_2015/'

# 合并文件
for filename in file:
    if not filename.startswith('.') or filename.endswith('.txt'):
        filedir2=os.listdir(f+filename)
        newname=os.getcwd()+'/sorted/'+filename+'.txt'
        # print(newname)
        newfile = open(newname, 'w')
        for i in filedir2:
            portion=os.path.splitext(filename)
            # print(portion)
            filepath=f+str(filename)+'/'+i
            print(filepath)
            for line in open(filepath):
                newfile.writelines(line)
            newfile.write('\n')
        newfile.close()


        # if portion[1]==".dat":
            #     newname=portion[0]+".txt"
            #     os.rename(filename,newname)




