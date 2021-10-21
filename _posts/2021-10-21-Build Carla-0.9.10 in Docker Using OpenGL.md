---
title: Build Carla-0.9.10 in Docker Using OpenGL 
layout: post
categories: Carla
tags: Carla
date: 2021-10-21 17:00
---
1. Requirements

   - Ubuntu 16.04+ 64-bit version;
   - Minimum 8GB of RAM;
   - Minimum of 300GB available disk space;
   - Python 3.6+;
   - Docker;
   - nvidia-docker;
   - OpenGL or Vulkan graphics driver;

2. Dependencies
   ```
   pip3 install ue4-docker
   ```

   这是一个构建carla镜像所依赖的镜像。如果要自动配置Linux防火墙或检测此镜像是否通信正常，则执行：

   `ue4-docker setup`

   如果出现`No firewall configuration required`则正常。

   如果在安装过程中出现需要升级pip以及安装testresources的问题，执行：

   `pip3 install --upgrade pip`

   `sudo apt install python-testresources`

3. 构建ue4的镜像

   将想要构建的某特定版本的carla源码从https://github.com/carla-simulator/carla/tree/0.9.10下载下来（也可使用git clone下载后进行版本回退，参考步骤10），我下载的是0.9.10版本，解压后将当前工作目录切换到carla/Util/Docker/目录下，执行：

   `ue4-docker build 4.24.3 --no-engine --no-minimal`

   可以通过在终端显示的信息看到，在构建镜像时，运行到step14的时候可能会出现如下问题：

   ```
   Cloning into '/home/ue4/UnrealEngine'...
   fatal: unable to access 'https://github.com/EpicGames/UnrealEngine.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
   ```

   这是因为自己的Github账户和UnrealEngine没有进行链接的缘故。因为UnrealEngine的repo是private的，如果你的github没有连接到一起，是无法下载引擎Repo的。因此需要先注册一个Github帐号，然后再注册一个UnrealEngine的帐号，将两个账户进行关联。具体的步骤如下：

   ![image-20211019162340983](/home/zhangyuan/.config/Typora/typora-user-images/image-20211019162340983.png)

   然后安装git：

   `sudo apt install git`

   进行了以上步骤后重新执行`ue4-docker build 4.24.3 --no-engine --no-minimal`，但仍有可能卡在step14上使得构建镜像无法成功。阅读出现问题的ue4-source的dockerfile后，发现在此dockerfile中会尝试使用三种方式来获取ue4源码：从上下文目录拷贝、通过构建密匙验证来获取、通过终端提供的密匙来获取。在前两种方式均失败（正常情况下前两种方式也确实是失败的）的情况下就会通过本人的git账户和ue4链接后自动获取的密匙来对git仓库进行克隆，但由于github网站从2021年8月开始了双重验证等一系列复杂的密保验证（参考https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/），以及国内的网络问题，也有极大的可能克隆失败。此时可能可以通过配置代理的方式来构建成功，但我使用的是下载源码并修改Dockerfile来将源码拷贝进容器来构建的，也的确构建成功了。配置代理的方式放在本文末尾进行说明。现在说明如何通过修改Dockerfile的方式来构建镜像。

   首先将UnrealEngine 4.24.3的仓库在https://github.com/EpicGames/UnrealEngine/tree/4.24.3-release下载下来并解压，修改文件夹名为UnrealEngine。为了找到ue4-source的Dockerfile，要安装一个文件搜索工具，fsearch和locate均可，fsearch具有GUI界面，安装方式如下：

   ```
   sudo add-apt-repository ppa:christian-boxdoerfer/fsearch-daily
   sudo apt update
   sudo apt install fsearch-trunk
   ```

   安装locate的方式如下：

   ```
   apt-get install mlocate
   //然后就可以进行查找文件了
   locate ue4-source
   ```

   找到ue4-source的dockerfile，打开此文件，发现step14对应的语句为：

   ```
   # Clone the UE4 git repository using the endpoint-supplied credentials
   RUN git clone --progress --depth=1 -b $GIT_BRANCH $GIT_REPO /home/ue4/UnrealEngine
   ```

   注释掉RUN...那条语句，添加一条COPY语句，则变为：

   ![image-20211019190858771](/home/zhangyuan/.config/Typora/typora-user-images/image-20211019190858771.png)

   将之前解压好的UnrealEngine文件夹放到此dockerfile所在的目录中，保存此dockerfile后重新执行`ue4-docker build 4.24.3 --no-engine --no-minimal`，则可以成功构建ue4-source，出现两个镜像，分别是ue4-source:4.24.3和ue4-source:4.24.3-opengl。

4. 构建carla-prerequisites镜像

   同样地，当前工作目录依然在carla/Util/Docker/目录下。阅读Prerequisites.Dockerfile，发现语句`FROM adamrehn/ue4-source:${UE4_V}-opengl`，意思是要构建的prerequisites镜像的基础镜像为上一步构建好的ue4-source:4.24.3-opengl。执行以下命令构建镜像prerequisites：

   ```
   docker build -t carla-prerequisites -f Prerequisites.Dockerfile .
   ```

   **注意不要遗漏最后那个点！**运行的时候可能会报如下几个错：

   ```
   // 1. apt-utils
   debconf: delaying package configuration, since apt-utils is not installed
   // 这个可以不用管
   
   // 2. gpg
   gpg: failed to start agent '/usr/bin/gpg/agent' : No such file or directory
   gpg: can't connect to the agent: No such file or directory
   // 这个好像没管也没啥问题
   
   // 3.pip3安装包时连接超时的问题
   ```

   第三个问题的具体报错信息忘记了，总归是要接着修改一下Prerequisites.Dockerfile，这个文件的目录为/carla仓库下载后解压的路径/Util/Docker/Prerequisies.Dockerfile。找到如下两行：

   ```
   pip3 install -Iv setuptools==47.3.1 && \
   pip3 install distro && \
   ```

   将其改为：

   ```
   pip3 install --default-timeout=1000 -Iv setuptools==47.3.1 && \
   pip3 install --default-timeout=1000 distro && \
   ```

   再重新执行`docker build -t carla-prerequisites -f Prerequisites.Dockerfile .`，可以看到构建成功，镜像名为carla-prerequisites:latest。

5. 最后构建真正的carla镜像

   执行如下命令：

   ```
   docker build -t carla -f Carla.Dockerfile .
   ```

   **注意不要遗漏最后那个点！**在step4时应该还是会因为代理的问题而出现：

   ```
   Cloning into 'carla'...
   fatal: unable to access 'https://github.com/carla-simulator/carla.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
   ```

   理论上应该可以通过配置代理的方法来解决。我依然是下载了对应0.9.10版本的carla源码，注意，此处直接把.zip文件放入此Carla.Dockerfile所在的目录中，修改Carla.Dockerfile，将：

   ```
   RUN cd /home/ue4 && \
     if [ -z ${GIT_BRANCH+x} ]; then git clone --depth 1 https://github.com/carla-simulator/carla.git; \
     else git clone --depth 1 --branch $GIT_BRANCH https://github.com/carla-simulator/carla.git; fi && \
     cd /home/ue4/carla && \
     ./Update.sh && \
     make CarlaUE4Editor && \
     make PythonAPI && \
     make build.utils && \
     make package && \
     rm -r /home/ue4/carla/Dist
   ```

   修改为：

   ```
   # RUN cd /home/ue4 && \
   #   if [ -z ${GIT_BRANCH+x} ]; then git clone --depth 1 https://github.com/carla-simulator/carla.git; \
   #   else git clone --depth 1 --branch $GIT_BRANCH https://github.com/carla-simulator/carla.git; fi && \
   
   COPY carla.zip /home/ue4/carla.zip
   RUN cd /home/ue4 && \
     unzip carla.zip && \
     cd /home/ue4/carla && \
     ./Update.sh && \
     make CarlaUE4Editor && \
     make PythonAPI && \
     make build.utils && \
     make package && \
     rm -r /home/ue4/carla/Dist
   ```

   保存后重新执行`docker build -t carla -f Carla.Dockerfile .`，执行到`./Update.sh`这个命令时会发现要下载一个12GB的东西，速度会非常慢，因此需要配置代理，方法在文章末尾部分。为了使用配置好的代理，我将直接run一个carla-prerequisite的容器，在里面执行carla.Dockerfile里面的命令。因此上面修改Dockerfile的过程是不必要的（看到这里也不要打我！）。

   配置好代理后，将之前手动下好的源码从主机复制到容器中（或者参考步骤10在容器中进行git clone下载后进行版本回退），在主机中执行：

   `docker cp ~/download/carla.zip [容器名]:/home/ue4/carla.zip`

   然后进入容器：

   `docker exec -it [容器名] /bin/bash`

   在容器中执行：

   ```
   su ue4
   unzip ~/carla.zip
   cd ~/carla
   ./Update.sh
   // 这一步需要相当久的时间，如果之前配置好代理了，就应该没问题了
   make CarlaUE4Editor
   make PythonAPI
   make build.utils
   make package
   rm -r ~/carla/Dist
   ```

6. 安装其余依赖（官方教程中没给出的但实测必须的）

   在容器中执行：

   ```
   pip3 install pygame numpy networkx
   exit //这一步是为了切换到root用户
   apt-get update
   apt-get install fontconfig libxrandr-dev xserver-xorg-dev xorg-dev mlocate
   su ue4
   ```

7. 测试UE4Editor

   在主机中将之前配置的代理文件删掉或改名，因为如果不这么做的话，UE4-Docker会无法与Carla的容器进行联合仿真，在主机中执行：

   ```
   mv ~/.docker/config.json ~/.docker/config.json.back
   ```

   在容器中执行：

   ```
   cd ~/UnrealEngine/Engine/Binaries/Linux 
    ./UE4Editor -opengl
    // -opengl参数一定不能略去
   ```

   可能会得到报错：

   `Trying to force OpenGL RHI but the project does not have it in TargetedRHIs list `

   这是因为我们在主机上安装的是OpenGL Graphics Driver，并且构建的也是使用OpenGL的ue。而UE4Editor的BaseEngine.ini文件中关于驱动的默认语句为Vulkan，因此我们要修改此文件，执行：

   ```
   vim ~/UnrealEngine/Engine/Config/BaseEngine.ini
   ```

   找到`+TargetedRHIs=SF_VULKAN_SM5`这一条语句，注释掉它，再将`+TargetedRHIs=GLSL_430`这一条去掉其注释符号。

   再重新执行`./UE4Editor -opengl`，可能会得到报错：

   ```
   > [ 0]LogModuleManager: Warning: ModuleManager: Unable to load module 'MegascansPlugin' - 0 instances of that module name found.
   // 此处的模块名可能是ue4的任意模块名
   ```

   这个报错我在网上找了许多方案，都不能解决我的问题，虽然报的是找不到模块的错，但是经过排查其实存放模块的文件夹内都好好的存在着那些被报错的模块。即使是重新编译ue4，也只能打开UE4Editor一次，关掉窗口后就不能再次打开了。我最后经过很长时间的尝试终于把这个bug解决了，现在说明解决方案。要使得已经编译安装的插件可被UE4Editor找到，必须满足两个条件：1. 插件的UE4Editor.modules中的BuildID必须正确，以ExampleDeviceProfileSelector模块为例，此模块的UE4Editor.modules的所在路径为`~/UnrealEngine/Engine/Plugins/Runtime/ExampleDeviceProfileSelector/Binaries/Linux/UE4Editor.modules`；2. 路径`~/UnrealEngine/Engine/Binaries/Linux`中必须包含模块对应的.debug、.so、.sym文件，以ExampleDeviceProfileSelector模块为例，它在上述路径中必须包含其对应的libExampleDeviceProfileSelector.debug、libExampleDeviceProfileSelector.so和libExampleDeviceProfileSelector.sym文件。这两个条件不满足则UE4Editor必不可能找到相应模块。

   因此，要满足第一个条件，我们要做的就是先找到任意一个没有问题的模块，假设没有问题的模块名为LauncherChunkInstaller，其所在路径为`~/UnrealEngine/Engine/Plugins/Portal/LauncherChunkInstaller`，则执行：

   ```
   vim ~/UnrealEngine/Engine/Plugins/Portal/LauncherChunkInstaller/Binaries/Linux/UE4Editor.modules
   ```

   打开后会显示：

   ```
   {
           "BuildId": "198ed7c9-1f6b-4944-b545-b50a5ad15ae5",
           "Modules":
           {
                   "ExampleDeviceProfileSelector": "libUE4Editor-LauncherChunkInstaller.so"
           }
   }
   // BuildId可能会根据编译版本或其他因素而不同，要看自己的模块的具体Id
   ```

   复制BuildId后面那串代码。假设有问题的模块为ExampleDeviceProfileSelector，其UE4Editor.modules所在路径为`~/UnrealEngine/Engine/Plugins/Runtime/ExampleDeviceProfileSelector/Binaries/Linux/UE4Editor.modules`，则执行：

   ```
   vim ~/UnrealEngine/Engine/Plugins/Runtime/ExampleDeviceProfileSelector/Binaries/Linux/UE4Editor.modules
   ```

   会发现它的BuildId和能正确找到的模块的BuildId是不同的，因此我们将它的BuildId替换为之前复制的那串正确的BuildId，然后保存后退出。接着要满足第二个条件，执行：

   ```
   cp ~/UnrealEngine/Engine/Plugins/Runtime/ExampleDeviceProfileSelector/Binaries/Linux/libUE4Editor-ExampleDeviceProfileSelector.* ~/UnrealEngine/Engine/Binaries/Linux
   ```

   对于所有未找到的模块都使用以上解决方案即可。然后重新执行：

   `./UE4Editor -opengl`

   此时打开它应该没什么问题了，可能会报：

   `Warning: OpenGL is dep`recated, please use Vulkan

   直接点OK即可，因为我主机上本来就没安装Vulkan。如果这个时候再安装Vulkan相关组件的话，那么可能之前的那些构建工作都白做了，千万别弄那些。第一次打开UE4Editor可能需要蛮长时间，后面就会快很多了，无报错打开后就是这个界面：

   ![image-20211020112803612](/home/zhangyuan/.config/Typora/typora-user-images/image-20211020112803612.png)

   选中CarlaUE4，点击Open Project，可能会提示此项目与当前版本不符，问你要不要复制一个，之类的问题，点确定即可。然后又会进入漫长的等待。正确打开后就会呈现这样的界面：

   ![image-20211020114130890](/home/zhangyuan/.config/Typora/typora-user-images/image-20211020114130890.png)

   关闭此界面。如果出现了其他报错，那么请确认自己主机在安装图形驱动时也同时安装了OpenGL，因为据我观察，国内的许多安装Nvidia Driver的教程中，安装命令通常都带了--no-opengl参数，如果带了这个参数那么我建议重新安装Nvidia Driver。如果还出现了其他报错，那么通常可以在Google上搜索到相应解决方案。

   **附加说明：如果进行以上debug操作后，UE4Editor依旧无法运行成功，那么可以重新编译UnrealEngine。**以下为重新编译UE4的操作（请务必保证已经尽可能debug了）：

   ```
   cd ~/UnrealEngine
   make
   ```

   编译结束后，再进行测试。

8. 测试示例程序

   在这一步我们对Carla的官方示例程序进行测试，如果能够显示成功则证明Carla的容器与ue4-docker能够成功地进行联合仿真。在容器中执行：

   ```
   cd ~/carla/Unreal/CarlaUE4\ 4.24/
   // 此处CarlaUE4 4.24是之前UE4Editor复制出的适应其版本的carlaUE4的项目文件夹
   ~/UnrealEngine/Engine/Binaries/Linux/UE4Editor "$PWD/CarlaUE4.uproject" -opengl
   //这条命令的意思是通过UE4Editor打开当前工作目录中的CarlaUE4.uproject，驱动指定为OpenGL
   ```

   此时会再次弹出第7步打开的那个界面，只不过现在打开的命令会是以后打开CarlaUE4的常规命令。打开仿真界面后，点击Play图标。不要关闭仿真界面，再起一个容器的终端，在里面执行：

   ```
   su ue4
   cd ~/carla/PythonAPI/examples
   python3 automatic_control.py
   ```

   如果一切正常，那么应该会弹出一个Pygame窗口，正常运行。Build Carla-0.9.10 in Docker Using OpenGL到此结束！

9. 配置镜像源和代理

   首先配置代理，在主机的终端中执行：

   ```
   touch ~/.docker/config.json
   vim ~/.docker/config.json
   ```

   在打开的文件中写入：

   ```
   {
   		 "proxies":
   		 {
    		  "default":
   		   {
        		"httpProxy": "http://127.0.0.1:2340",
     		   "httpsProxy": "http://127.0.0.1:2340",
        		"noProxy": "localhost"
   		   }
   		 }
   }
   // http://127.0.0.1:2340根据自己的主机ip和端口号进行替换
   ```

   然后执行如下命令，`--net=host`是将容器和主机使用同一个网络命名域：

   ```
   docker run -e DISPLAY=$DISPLAY --net=host --gpus all --runtime=nvidia --user 0 --name [容器名] -it carla-preprequisite:latest /bin/bash
   ```

   进入后的当前用户为root。在容器中的终端执行

   ```
   apt-get update
   apt-get install vim
   ```

   在主机的终端中执行如下命令，将容器中的镜像源添加阿里云镜像源：

   ```
   docker cp /etc/apt/sources.list.d/ali.list [容器名]:/etc/apt/sources.list/ali.list
   ```

   然后使用Crtl+p+q暂停容器。

10. 使用git clone下载仓库并进行版本回退

    以Carla-0.9.10为例，首先执行以下命令将仓库下载到当前工作目录中：

    ```
    git clone -b master https://github.com/carla-simulator/carla
    // 这里使用-b master指定下载master分支的仓库
    ```

    然后在https://github.com/carla-simulator/carla/tags中找到0.9.10版本对应的commit编号c7b2076：

    ![image-20211021151629578](/home/zhangyuan/.config/Typora/typora-user-images/image-20211021151629578.png)

    然后执行以下命令：

    ```
    cd [clone下来的项目文件夹路径]
    git checkout c7b2076
    // git checkout 后接项目版本对应的commit编号
    ```

    
