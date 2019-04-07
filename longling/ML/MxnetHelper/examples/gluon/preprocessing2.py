#!/user/bin/python
# -*- coding: UTF-8 -*-
import operator as op

# 由graph图建立索引
# 由graph建立数字化的图
graphdic={}
the_graph=[]
g=open("gzsx.post","r")
index=open("the_index_of_node","w")
graph=open("the_graph","w")
line=g.readline()
tmp=line.split()
graphdic.update({tmp[0]:0})
graphdic.update({tmp[2]:1})
index.write(tmp[0]+' '+str(0)+'\n')
index.write(tmp[2]+' '+str(1)+'\n')
i=2
the_graph.append([0,1])
j=1
graph.write('0'+' '+'1'+'\n')
while line:
    line = g.readline()
    tmp = line.split()
    if op.eq(line,''):
        break
    else:
        if tmp[0] not in graphdic:
            graphdic.update({tmp[0]:i})
            index.write(tmp[0]+' '+str(i)+'\n')
            print("i:",i)
            i=i+1
        the_pre = graphdic[tmp[0]]

        if tmp[2] not in graphdic:
            graphdic.update({tmp[2]:i})
            index.write(tmp[2]+' '+str(i)+'\n')
            print("i:",i)
            i=i+1
        the_post=graphdic[tmp[2]]

        the_graph.append([the_pre,the_post])
        j=j+1
        print(j)
print(i)
the_graph.sort()
for ele in the_graph:
    graph.write(str(ele[0])+' '+str(ele[1])+'\n')
graph.close()

