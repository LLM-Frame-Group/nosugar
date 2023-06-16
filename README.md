# 介绍

本仓库用于存储博客网站框架，可以使用此框架上传文章到博客网站。

# 环境配置

## 安装nodejs

1. 安装nodejs

   如果没有安装nodejs，请自行[安装](https://nodejs.org/en/download)

   使用`node --version`验证安装效果

   如果没有出现版本号，则手动配置一下配置环境变量，直接把安装后Nodejs的根目录加入到Path即可。

2. 安装hexo

   ```
   npm install -g hexo-cli 
   ```

## 第一次部署

如果首次clone仓库，需要安装相关module

```
npm install
```



# 上传博客

## 新建

可以使用命令新建一个新markdown文档

```
hexo n "你的文章名"
```

当然，也可以把现有的md放到`\Blog\source\_posts`，再手动建一个同名assets文件夹，效果是一样的。

**注意，非hexo创建的md文档需要在头部插入一段header（从其他文档里复制一个）确保正确渲染**

## 编辑

在 `\source\_posts`中可以找到刚刚新建的md文件，以及一个同名的assets文件夹

### 文章简介

其中md文件的头部有一段注释，里面的内容指定了文档的渲染设置

常规模板：

```
title: 文章标题
date: y-m-d hh:mm:ss
tags: 标签（可以通过[tag1,tag2,...]的形式添加多个标签）
excerpt: 文章摘要（不写会放全文）
mathjax: true
author: 作者名称
```

其他header

```
subtitle:副标题
```



注意，其中的冒号是英文冒号，冒号后面必须跟一个空格才能输入内容



### 图片格式

正常的markdown格式，但插入的图片格式略有不同，有2种方法插入图片

1. 使用网络图片，找个图床，直接插入图片链接

2. 使用本地图片，图片需要放在和文章一起创建的文件夹，然后以下面的格式进行引用（就是正常的md图片引用格式）

   ```
   ![图片标题](./assets文件夹名/图片名.png)
   ```



## 渲染

更新文件

```
hexo g
```



本地查看

```
hexo s
```



## 上传

确认内容无误即可回到根目录，将更新上传到nosugar仓库

```
git pull
git add .
git commit -m "update info"
git push
```

仓库会自动把内容更新到博客仓库，过程会有一定延迟，可以在nosuar仓库的action界面查看进度
