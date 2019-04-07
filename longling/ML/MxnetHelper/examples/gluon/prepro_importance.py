#!\user\bin\python
# -*- coding:UTF-8 -*-

import operator as op

# 生成结点（编号形式）对应的重要度 没有的则设为0
# 更新不在图中的index

f_nodeimportance=open("the_index_of_node","r")
nodeim=open("the_importance","w")
wholeim=open("the_all_importance","w")

nodedict={}
line=f_nodeimportance.readline()
tmp=line.split()
i=0
while line:
    if op.eq(line,''):
        break
    else:
        nodedict.update({tmp[0]:int(tmp[1])})
        line=f_nodeimportance.readline()
        tmp=line.split()
        i=i+1
print(i)
num_node=i
f_nodeimportance.close()

importance_of_node=[]
file_name=["0204","0205","0504","0505","0604","0605"]
for ele in file_name:
    f=open(ele,"r")
    line = f.readline()
    tmp = line.split()
    while line:
        if op.eq(line,''):
            break
        else:
            if tmp[0] in nodedict:
                importance_of_node.append([nodedict[tmp[0]],float(tmp[1])])
                wholeim.write(str(nodedict[tmp[0]])+' '+tmp[1]+'\n')
            else:
                wholeim.write(str(i)+' '+tmp[1]+'\n')
                nodedict.update({tmp[0]:i})
                i=i+1
            line=f.readline()
            tmp=line.split()
    f.close()
print(i)
wholeim.close()

importance_of_node.sort()
k=0
# 缺失的index重要度设为0
for i in range(num_node):
    if not op.eq(importance_of_node[i][0],k):
        the_gap=importance_of_node[i][0]
        while (k<the_gap):
            importance_of_node.append([k,0])
            k=k+1
    k=k+1
importance_of_node.sort()
for node in importance_of_node:
    nodeim.write(str(node[0])+' '+str(node[1])+'\n')
nodeim.close()

# 构建一个新的index表 储存nodedict
new_index=open("newindex","w")
for key in nodedict.keys():
    new_index.write(key+' '+str(nodedict[key])+'\n')
new_index.close()