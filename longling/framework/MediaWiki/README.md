# MeidaWiki 安装
*位置: 172.16.46.213*
*操作系统: CentOS Linux release 7.4.1708 (Core)*
*MediaWiki版本:  1.31.0*
## 环境配置
### 数据库
#### [PostgreSQL](https://www.postgresql.org/download/)
使用下列命令来查看PostgreSQL是否安装
```sh
psql
```
如果没有，按下面方式安装PostgreSQL(可能需要管理员权限)
```sh
yum install https://download.postgresql.org/pub/repos/yum/10/redhat/rhel-7-x86_64/pgdg-centos10-10-2.noarch.rpm
yum install postgresql10
yum install postgresql10-server
/usr/pgsql-10/bin/postgresql-10-setup initdb
systemctl enable postgresql-10
systemctl start postgresql-10
```
开启5432端口
```sh
sudo firewall-cmd --zone=public --permanent --add-port=5432/tcp
```
安装结束后，以超级用户 postgre 登录
```sh
sudo su - postgres
psql
```
执行下列命令
```sql
CREATE USER wikiuser WITH NOCREATEDB NOCREATEROLE NOSUPERUSER ENCRYPTED PASSWORD 'password';
CREATE DATABASE wikidb WITH OWNER wikiuser;
```
#### MySQL
进入mysql服务器
```sh
mysql -u root -h 172.16.46.203 -p
```
创建数据库
```sql
CREATE DATABASE wikidb;
GRANT ALL PRIVILEGES ON wikidb.* TO 'wikiuser'@'172.16.46.213' IDENTIFIED BY 'lab502';
```
测试
```sh
mysql -u wikiuser -h 172.16.46.203 -plab502 -D wikidb
```
### Apache
直接浏览器中访问 localhost，如果可以访问，证明已安装Apache
否则，使用
```sh
sudo systemctl status httpd.service
```
查看是否有服务，有的话可能是由于防火墙的关系。
管理员权限下开启防火墙80端口
```sh
firewall-cmd --zone=public --permanent --add-port=80/tcp
firewall-cmd --reload
```
否则，按下列方式安装(可能需要管理员权限)
```sh
yum -y install httpd
systemctl start httpd.service
systemctl enable httpd.service
```
### PHP
查看是否已经安装PHP
```sh
php -v
```
如果没有，按下列方式安装php(可能需要管理员权限)
```sh
yum -y install php
```
*注意*
如果版本不是 7 以上的话需要更新源
按下列方式进行(需要管理员权限)
```sh
sudo yum install epel-release yum-utils
sudo yum install http://rpms.remirepo.net/enterprise/remi-release-7.rpm
sudo yum-config-manager --enable remi-php72
sudo yum install php php-common php-opcache php-mcrypt php-cli php-gd php-curl php-mysql php-mbstring php-xml
```
修改后需要重启Apache
```sh
sudo service httpd restart
```
**如果是使用postgreSQL的话**
还需要安装php的postgreSQL支持
```sh
sudo yum install php-pgsql
```
**其它可选优化安装选项**
安装php-apcu缓存
```sh
sudo yum install php-apcu
```
安装php-intl支持
```sh
sudo yum install php-intl
```
**记得php更改后要重启Apache**
```sh
sudo service httpd restart
```

## 通过php安装MediaWiki
[下载](https://www.mediawiki.org/wiki/Manual:Installing_MediaWiki/zh) Mediawiki 并解压到 /var/www/html/w (可能需要管理员权限)

则通过浏览器来访问
```vim
172.16.46.213/w/index.php
```

可能出现 *Forbidden* 错误

### 连接数据库

标识本wiki:

数据库名称: wikidb

MediaWiki的数据库模式: MediaWiki版本

数据库用户名：postgres

数据库密码：%postgres的用户密码%

可能出现 *5432* 错误

### 


### 常见问题列表
#### Forbidden
Err:
通过浏览器访问index.php, 发现**Forbidden**无法访问

Sol:

修改权限
```sh
chown -R 755 /var/www/html/w/
chmod 755 /var/www/html/w/mw-config
```
还不行的话
查看
```sh
getenforce
```
是否为 "Enforcing"
如果是的话，则是由 SELinux导致的
关掉就可以了
```sh
sudo setenforce 0
```
改一下类型
```sh
sudo chcon -Rv --type=httpd_t /var/www/html/w
```
改完把SELinux恢复一下
```sh
sudo setenforce 1
```

#### 5432
Err:
Cannot access the database: pg_connect(): Unable to connect to PostgreSQL server: could not connect to server: Permission denied Is the server running on host "localhost" (::1) and accepting TCP/IP connections on port 5432? could not connect to server: Permission denied Is the server running on host "localhost" (127.0.0.1) and accepting TCP/IP connections on port 5432?

Sol:
```sh
sudo setsebool -P httpd_can_network_connect_db on
```
