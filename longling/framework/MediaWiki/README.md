# MeidaWiki 安装
*位置: 172.16.46.213*
*操作系统: CentOS Linux release 7.4.1708 (Core)*
*MediaWiki版本:  1.31.0*
## 环境配置
### 数据库
*以 [PostgreSQL](https://www.postgresql.org/download/) 为例*
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

## 通过php安装MediaWiki
[下载](https://www.mediawiki.org/wiki/Manual:Installing_MediaWiki/zh) Mediawiki 并解压到 /var/www/html/w (可能需要管理员权限)

则通过浏览器来访问
```vim
172.16.46.213/w/index.php
```
如果发现**Forbidden**无法访问 ，修改权限
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