# -*- coding:utf-8 -*-
# Description: Function Description
# Copyright: Copyright (c) 2022
# Company: Ruijie Co., Ltd.
# Create Time: 2022年04月25日
# @author huangtinghong
from distutils.core import setup
import setuptools

setup(name='distribute-train',  # 打包后的包文件名(注意:这里是pip list中显示的名字 并不是import的名字)
      version='0.1.0',  # pip中显示的版本号
      description='分布式训练工具',  # 项目描述
      author='huangtinghong',  # 作者名字
      packages=setuptools.find_packages()  # 打包的包真正要import的包 需要创建文件夹里面必须有__init__.py(可为空)
      )