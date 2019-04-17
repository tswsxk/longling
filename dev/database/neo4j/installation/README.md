# Neo4j 部署指南
*社区版 3.4*
## [安装](https://neo4j.com/download/other-releases/#releases "installation tutorial")
### [CentOS](http://yum.neo4j.org/stable/?_ga=2.140724145.202350270.1530342739-105396060.1530342739 "yum tutorial")
获取Neo4j 的 yum repo key
```sh
cd /tmp
wget http://debian.neo4j.org/neotechnology.gpg.key
```
导入repo key（可能需要管理员权限）
```sh
rpm --import neotechnology.gpg.key
```
向/etc/yum.repos.d/neo4j.repo添加下面内容（可能需要管理员权限）
```vim
[neo4j]
name=Neo4j Yum Repo
baseurl=http://yum.neo4j.org/stable
enabled=1
gpgcheck=1
```
之后yum安装（可能需要管理员权限）
```sh
yum install neo4j
```
*如果安装过程中出现了*
```sh
https://repo.mongodb.org/yum/redhat/7/mongodb-org/3.4/x86_64/repodata/repomd.xml: [Errno 12] Timeout on https://repo.mongodb.org/yum/redhat/7/mongodb-org/3.4/x86_64/repodata/repomd.xml: (28, 'Operation timed out after 30000 milliseconds with 0 out of 0 bytes received')
Trying other mirror.
```
*不要方，后面能找到源*

**出现这个的时候一定输入y，默认是不安装的选项**
```sh
Is this ok [y/d/N]: y
```
### 安装验证
安装成功提示
```sh
Complete!
```
```sh
service neo4j status
```
通过service命令查看出现
```sh
● neo4j.service - Neo4j Graph Database
   Loaded: loaded (/usr/lib/systemd/system/neo4j.service; disabled; vendor preset: disabled)
   Active: inactive (dead)
```
证明安装成功，且.service文件创建成功，可以进行服务化部署步骤
## 服务化部署
以下所有步骤可能都需要管理员权限
### 简单启动
```sh
service neo4j start
```
### 灵活配置
参见[官方文档](https://neo4j.com/docs/operations-manual/3.2/installation/linux/systemd/)
### 开启对外服务接口
修改 /etc/neo4j/neo4j.conf文件 (可能需要管理员权限)
找到这个配置项
```vim
#dbms.connectors.default_listen_address=0.0.0.0
```
去掉 # 即可
其它可选配置项修改
以下三种协议任意开启一个即可（初始化全部关闭，至少开启一个，系统默认bolt连接，建议开启bolt）
```sh
dbms.connector.bolt.listen_address=:7687
```
```sh
dbms.connector.http.listen_address=:7474
```
```sh
dbms.connector.https.listen_address=:7473
```
然后重启服务
如果修改后仍然无法访问，可能是由防火墙拦截导致，参考[防火墙导致无法访问](#### 防火墙导致无法访问)来开启对应端口
#### 防火墙导致无法访问
*以下命令可能需要管理员权限*

*以http为例，端口号7474*

通过命令
```sh
firewall-cmd --list-ports
```
查看对应端口是否开启

如果没有，通过命令
```sh
firewall-cmd --zone=public --permanent --add-port=7474/tcp
firewall-cmd --reload
```
开启端口

## 使用
### 默认
使用浏览器访问 http://localhost:7474/
外部访问用服务器ip地址替换 localhost 即可
**外部访问之前需要开启对外服务接口**
**初始化密码 neo4j，首次登陆后记得修改**
## Drivers配置
### Python

