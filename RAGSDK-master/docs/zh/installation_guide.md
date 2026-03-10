# 安装部署<a name="ZH-CN_TOPIC_0000002018714781"></a>

## 安装说明<a name="ZH-CN_TOPIC_0000002018595369"></a>

RAG SDK支持在容器内部署和在物理机内部署两种安装方式。

-   容器化部署流程如[图1](#fig66411525144113)所示，具体方法请参见[容器内部署RAG SDK](./installation_guide.md#容器内部署rag-sdk)。
-   物理机内部署流程如[图2](#fig188855012335)所示，具体方法请参见[物理机内部署RAG SDK](./installation_guide.md#物理机内部署rag-sdk)。

**图 1** RAG SDK容器化部署流程<a id="fig66411525144113"></a>  
![](figures/RAG-SDK容器化部署流程.png "RAG-SDK容器化部署流程")

**图 2** RAG SDK物理机内部署流程<a id="fig188855012335"></a>  
![](figures/RAG-SDK物理机内部署流程.png "RAG-SDK物理机内部署流程")


## 获取RAG SDK软件包<a name="ZH-CN_TOPIC_0000001982155320"></a>

**获取软件包<a name="section927075392218"></a>**

请参考本章获取所需软件包和对应的数字签名文件，下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

|组件名称|软件包|获取链接|
|--|--|--|
|RAG SDK|检索增强软件包|<a href="https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann">获取链接</a>|


> [!NOTE] 说明
>容器内自带有RAG SDK，如果需要更新，请参见[升级](#升级)。

**软件数字签名验证<a name="section1269731820717"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)


## 安装依赖<a name="ZH-CN_TOPIC_0000002018714993"></a>

为保证RAG SDK的正常使用，需要安装相关依赖。

-   如果在物理机内部署RAG SDK，需要安装[表1](#table285894914124)中的所有依赖包。
-   如果在容器内部署RAG SDK，需要在宿主机上安装npu-driver驱动包、npu-firmware固件包和Ascend Docker Runtime，并启动MindIE推理服务，其他操作请参考[容器内部署RAG SDK](#容器内部署rag-sdk)。

**安装依赖<a name="section10458105983310"></a>**

1.  安装NPU驱动固件，详细步骤请参见《CANN 软件安装指南》中的“安装NPU驱动和固件”章节（商用版）或“安装NPU驱动和固件”章节（社区版）内容。为了让非root用户能够使用驱动，驱动的安装要添加**--install-for-all**选项。
2.  安装CANN Toolkit、ops和NNAL加速库，详细步骤请参见《CANN 软件安装指南》中的“安装依赖”和“安装CANN软件包”章节。建议以普通用户HwHiAiUser进行安装。如果使用AscendHub镜像部署RAG SDK，无需执行该步骤。
3.  安装并运行推理大模型，详细步骤请参见《MindIE安装指南》中的“方式三：容器安装方式”章节和“配置Server”章节。
4.  安装Ascend Docker Runtime，详细步骤请参见《MindCluster  集群调度用户指南》的“安装 \> 安装部署”章节。

**下载依赖软件包<a name="section107791548183117"></a>**

**表 1**  昇腾软件依赖
<a id="table285894914124"></a>

<table>
<tr>
<th>软件包简称</th>
<th>安装包全名</th>
<th>配套版本</th>
<th>获取链接</th>
</tr>

<tr>
<td>CANN软件包</td>
<td>Ascend-cann-toolkit_<em id="i193132091185"><a name="i193132091185"></a><a name="i193132091185"></a><span id="ph33611452313"><a name="ph33611452313"></a><a name="ph33611452313"></a>&lt;version&gt;</span></em>_linux-<em id="i18718152018815"><a name="i18718152018815"></a><a name="i18718152018815"></a><span id="ph4968818194110"><a name="ph4968818194110"></a><a name="ph4968818194110"></a>&lt;arch&gt;</span></em>.run</td>
<td rowspan="3">8.5.0</td>
<td rowspan="3">商用版：<a href="https://www.hiascend.com/developer/download/commercial/result?module=cann">获取链接</a><br>社区版：<a href="https://www.hiascend.com/developer/download/commercial/result?module=cann">获取链接</a></td>
</tr>
<tr>
<td>CANN算子包ops</td>
<td>Ascend-cann-<em id="i469981141315"><a name="i469981141315"></a><a name="i469981141315"></a>&lt;chip_type&gt;<span id="ph1969931151310"><a name="ph1969931151310"></a><a name="ph1969931151310"></a>-</span></em><span id="ph1670061151313"><a name="ph1670061151313"></a><a name="ph1670061151313"></a>ops</span>_<em id="i52841155578"><a name="i52841155578"></a><a name="i52841155578"></a>&lt;version&gt;</em>_linux-<em id="i1082013102713"><a name="i1082013102713"></a><a name="i1082013102713"></a>&lt;arch&gt;</em>.run</td>
</tr>
<tr>
<td>NNAL加速库（可选）</td>
<td>Ascend-cann-nnal_&lt;<em id="i43903543018"><a name="i43903543018"></a><a name="i43903543018"></a>version</em>&gt;_linux-&lt;<em id="i1274619579012"><a name="i1274619579012"></a><a name="i1274619579012"></a>arch</em>&gt;.run</td>
</tr>
<tr>
<td>npu-driver驱动包</td>
<td>Ascend-hdk-<em id="i1935205617"><a name="i1935205617"></a><a name="i1935205617"></a>&lt;chip_type&gt;</em>-npu-driver_<em id="i7935130468"><a name="i7935130468"></a><a name="i7935130468"></a>&lt;version&gt;</em>_linux-<em id="i9935706611"><a name="i9935706611"></a><a name="i9935706611"></a>&lt;arch&gt;</em>.run</td>
<td rowspan="2">25.5.0</td>
<td rowspan="2">商用版：<a href="https://www.hiascend.com/zh/hardware/firmware-drivers/commercial">获取链接</a><br>社区版：<a href="https://www.hiascend.com/hardware/firmware-drivers?tag=community">获取链接</a></td>
</tr>
<tr>
<td>npu-firmware固件包</td>
<td>Ascend-hdk-<em id="i173852861"><a name="i173852861"></a><a name="i173852861"></a>&lt;chip_type&gt;</em>-npu-firmware_<em id="i37315214614"><a name="i37315214614"></a><a name="i37315214614"></a>&lt;version&gt;</em>.run</td>
</tr>
<tr>
<td>Index SDK检索软件包</td>
<td>Ascend-mindxsdk-mxindex_<em id="i13839185810816"><a name="i13839185810816"></a><a name="i13839185810816"></a>&lt;version&gt;</em>_linux-<em id="i21954331981"><a name="i21954331981"></a><a name="i21954331981"></a>&lt;arch&gt;</em>.run</td>
<td>7.3.0</td>
<td><a href="https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann">获取链接</a></td>
</tr>
<tr>
<td>MindIE推理引擎软件包</td>
<td>Ascend-mindie_<em id="i97316211910"><a name="i97316211910"></a><a name="i97316211910"></a>&lt;version&gt;</em>_linux-<em id="i103261361689"><a name="i103261361689"></a><a name="i103261361689"></a>&lt;arch&gt;</em>.run</td>
<td>2.3.0</td>
<td><a href="https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann">获取链接</a></td>
</tr>
<tr>
<td>Ascend Docker Runtime</td>
<td>Ascend-docker-runtime_<em id="i11398712911"><a name="i11398712911"></a><a name="i11398712911"></a>&lt;version&gt;</em>_linux-<em id="i104571346888"><a name="i104571346888"></a><a name="i104571346888"></a><span id="ph163221324173214"><a name="ph163221324173214"></a><a name="ph163221324173214"></a>&lt;arch&gt;</span></em>.run</td>
<td>7.3.0</td>
<td><a href="https://gitcode.com/Ascend/mind-cluster/releases">获取链接</a></td>
</tr>
<tr>
<td>Python</td>
<td>-</td>
<td>3.11/3.12</td>
<td>请从<a href="https://gitcode.com/Ascend/mind-cluster/releases">Python官网</a>获取依赖软件</td>
</tr>
</table>

> [!NOTE] 说明
>-   <i><version\></i>表示软件版本号。
>-   <i><arch\></i>表示CPU架构。
>-   <i><chip\_type\></i>表示芯片类型。可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删除，即是<i><chip\_type\></i>的取值。
>-   为了让非root用户能够使用驱动，安装npu-driver要添加<b>--install-for-all</b>选项
>-   对于用户集成的开源和第三方软件，漏洞和问题请自行检查并及时进行修复；可以并且不限于通过[CVE（通用漏洞字典）官网](https://cve.mitre.org/cve/search_cve_list.html)确认对应开源软件版本的已知漏洞，并通过版本升级、使用patch补丁包更新等方式修复。


## 安装RAG SDK<a name="ZH-CN_TOPIC_0000002018595473"></a>

### 容器内部署RAG SDK<a name="ZH-CN_TOPIC_0000002018595365"></a>

本章节指导用户在容器内部署RAG SDK。

容器内运行RAG SDK的用户建议为普通用户。RAG SDK只支持基于Ubuntu基础镜像构建容器。

从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9)下载的镜像内已安装RAG SDK，如果需要更新，需要先卸载再安装，具体操作可参见[升级](#升级)。

**安装前准备<a name="section127111016173115"></a>**

-   已经根据[安装依赖](#安装依赖)章节安装所需的依赖。
-   配置源之前，请确保安装环境能够连接网络。

**操作步骤<a name="section15488921175211"></a>**

1.  获取基础镜像。
    -   推荐从昇腾镜像仓库获取RAG SDK镜像。步骤如下：
        1.  进入[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/b875f781df984480b0385a96fa1b03c9)。
        2.  单击“镜像版本”页签。
        3.  单击所需版本的“立即下载”按钮，按照页面提示进行下载。下载的镜像默认运行用户为普通用户“HwHiAiUser”。

    -   如果不从昇腾镜像仓库获取基础镜像，则用户自己准备一个镜像，步骤如下：
        1.  获取Dockerfile。Dockerfile中默认指定的安装和运行RAG SDK及相关依赖软件包的用户为普通用户“HwHiAiUser”，如果需要指定为其他用户，请在Dockerfile文件中适配修改。
        2.  在Dockerfile同级目录下创建package目录，并在package目录下存放[安装依赖](#安装依赖)章节中获取的依赖包，和[获取RAG SDK软件包](#获取rag-sdk软件包)章节中获取的RAG SDK软件包。

            文件存放结构如下示例，其中driver目录为npu-driver驱动安装后的目录，driver默认安装路径为“/usr/local/Ascend/driver/”。

            ```
            $ tree -L 2
            .
            |-- Dockerfile
            `-- package
                |-- Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run
                |-- Ascend-cann-toolkit_<version>_linux-<arch>.run
                |-- Ascend-cann-nnal_<version>_linux-<arch>.run
                |-- Ascend-mindxsdk-mxindex_<version>_linux-<arch>.run
                |-- Ascend-mindxsdk-mxrag_<version>_linux-<arch>.run
                `-- driver
            ```

        3.  执行以下命令，构建RAG SDK镜像。

            ```
            docker build -t <镜像名称>:<镜像tag> --build-arg http_proxy=<代理> --build-arg https_proxy=<代理> --build-arg ARCH=$(uname -m) --build-arg PLATFORM=<chip_type> -f Dockerfile .
            ```

            <i><chip\_type\></i>表示芯片名称，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删除，即是<i><chip\_type\></i>的取值。若是Atlas 800I A3 超节点服务器则取值为A3。

2.  运行RAG SDK。
    1.  执行如下命令运行RAG SDK容器。

        ```
        docker run -e ASCEND_VISIBLE_DEVICES=<device_id>  -itd --name <mxrag_demo> -p <port>:<port> -v <model_dir>:<model_dir>:ro <镜像名称>:<镜像tag> bash
        ```

        > [!NOTE] 说明 
        >-   <_device\_id_\>：表示NPU设备ID，默认从0开始。如果有多个，中间用英文","隔开。每个容器使用的NPU卡只能独占，否则会报错。可通过执行**ls /dev/davinci\* |grep -v /dev/davinci\_manager |tr -d /dev/davinci**命令查询。
        >-   <_mxrag\_demo_\>：表示运行后的容器名称，默认为mxrag\_demo。
        >-   <port\>：表示需要映射的端口。
        >-   <_model\_dir_\>：表示RAG SDK使用的模型存放的上级目录，如/home/data，不能配置为/home和/home/HwHiAiUser，防止宿主机文件挂载覆盖容器中HwHiAiUser家目录文件，导致RAG SDK功能异常。
        >-   对于Atlas 800I A2设备使用clip向量模型加速时，需添加--device=/dev/dvpp\_cmdlist:/dev/dvpp\_cmdlist:rw 参数支持对dvpp对图片预处理，并且需保证容器运行用户对/dev/dvpp\_cmdlist有访问权限。

    2.  进入容器。

        ```
        docker exec -it <mxrag_demo> bash
        ```

    3.  执行**npu-smi info**命令检查驱动是否挂载正常。

        如当Health参数的值为OK时，即表示当前芯片的健康状态为正常（以下仅为示例，请以实际查询到的信息为准）。

        ```
        +--------------------------------------------------------------------------------------------------------+
        | npu-smi 24.1.rc2                            Version: 24.1.rc2                                          |
        +-------------------------------+-----------------+------------------------------------------------------+
        | NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
        | Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
        +===============================+=================+======================================================+
        | 7       xxx                 | OK              | NA           44                0     / 0             |
        | 0       0                     | 0000:83:00.0    | 0            1851 / 21527                            |
        +===============================+=================+======================================================+
        | 8       xxx                 | OK              | NA           44                0     / 0             |
        | 0       1                     | 0000:84:00.0    | 0            1852 / 21527                            |
        +===============================+=================+======================================================+
        +-------------------------------+-----------------+------------------------------------------------------+
        | NPU     Chip                  | Process id      | Process name             | Process memory(MB)        |
        +===============================+=================+======================================================+
        | No running processes found in NPU 7                                                                    |
        +===============================+=================+======================================================+
        | No running processes found in NPU 8                                                                    |
        +===============================+=================+======================================================+
        ```


### 物理机内部署RAG SDK<a name="ZH-CN_TOPIC_0000001984017610"></a>

本章节指导用户基于操作系统ubuntu20.04-live-server和Huawei Cloud EulerOS 2.0，以普通用户“HwHiAiUser”为例，在物理机内部署RAG SDK。

-   确保已经根据[安装依赖](#安装依赖)章节安装所需的依赖。
-   安装RAG SDK和安装CANN的用户需为同一用户，建议为普通用户。

**安装前准备<a name="section189916619599"></a>**

-   基于操作系统ubuntu20.04-live-server，确保系统已安装Python 3.11，libpq-dev以及cmake，cmake最小版本为3.24.3。以下为libpq-dev和Python的安装方法：

    ```
    # 安装libpq-dev（psycopg2需要）
    apt install -y libpq-dev
    # 设置PY_VERSION为python3.11
    export PY_VERSION=python3.11
    # 添加Python ppa
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
    # 安装Python
    apt-get install -y --no-install-recommends $PY_VERSION $PY_VERSION-dev $PY_VERSION-distutils $PY_VERSION-venv
    # 设置默认的python
    ln -sf /usr/bin/$PY_VERSION /usr/bin/python3
    ln -sf /usr/bin/$PY_VERSION /usr/bin/python
    # 安装pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    python3 -m pip install --upgrade setuptools
    ```

-   基于操作系统Huawei Cloud EulerOS 2.0，Python及依赖包的安装方法：
<br># 确保系统yum源中包含python3.11，配置方法可参考[HCE的REPO源配置与软件安装](https://support.huaweicloud.com/usermanual-hce/hce_repo.html)的官方源配置和[13.12openEuler的REPO源配置](./1.faq.md#openeuler的repo源配置)

    ```
    yum update
    yum install python3.11
    yum install cmake swig postgresql-devel patch mesa-libGL
    ```

**操作步骤<a name="section539219178112"></a>**

1.  切换用户至HwHiAiUser，并进入“/home/HwHiAiUser”目录。
2.  安装torch和torch-npu。

    ```
    # for x86,安装torch:
    pip3 install torch==2.1.0+cpu  --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    # for aarch64,安装torch:
    pip3 install torch==2.1.0 
    # for all,安装torch-npu
    pip3 install torch-npu==2.1.0.post12
    ```

3.  安装torchvision-npu

    ```
    # 下载Torchvision Adapter代码，进入插件根目录
    git clone https://gitee.com/ascend/vision.git vision_npu
    cd vision_npu
    git checkout v0.16.0-6.0.0
    # 安装依赖库
    pip3 install -r requirement.txt
    # 配置cann环境变量
    source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
    # 编包
    python3 setup.py bdist_wheel
    # 安装
    cd dist
    pip3 install torchvision_npu-*.whl
    ```

4.  安装OpenBLAS。
    1.  下载OpenBLAS v0.3.10源码压缩包并解压。

        ```
        wget https://github.com/xianyi/OpenBLAS/archive/v0.3.10.tar.gz -O OpenBLAS-0.3.10.tar.gz
        tar -xf OpenBLAS-0.3.10.tar.gz
        ```

    2.  进入OpenBLAS目录。

        ```
        cd OpenBLAS-0.3.10
        ```

    3.  编译安装。

        ```
        make FC=gfortran USE_OPENMP=1 -j
        # 普通用户需要指定安装路径
        make PREFIX=/home/HwHiAiUser/OpenBLAS install
        ```

    4.  配置库路径的环境变量。

        ```
        vim ~/.bashrc
        # 在文件末添加如下信息
        export LD_LIBRARY_PATH=/home/HwHiAiUser/OpenBLAS/lib:$LD_LIBRARY_PATH
        ```

    5.  验证是否安装成功。

        ```
        cat /home/HwHiAiUser/OpenBLAS/lib/cmake/openblas/OpenBLASConfigVersion.cmake | grep 'PACKAGE_VERSION "'
        ```

        如果正确显示软件的版本信息，则表示安装成功。

5.  下载faiss源码，构建faiss wheel包并安装。

    > [!NOTE] 说明 
    >在安装Index SDK依赖时也安装了faiss，但仅编译生成了libfaiss.so，还需要构建faiss wheel包并安装，以便在python中使用faiss。

    1.  下载faiss源码包并解压。

        ```
        # faiss 1.10.0
        wget https://github.com/facebookresearch/faiss/archive/v1.10.0.tar.gz 
        tar -xf v1.10.0.tar.gz && cd faiss-1.10.0/faiss
        ```

    2.  创建install\_faiss.sh脚本。

        ```
        vi install_faiss.sh
        ```

    3.  在install\_faiss.sh脚本中写入如下内容。

        ```
        export FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.10.0
        # faiss安装后可能是${FAISS_INSTALL_PATH}/lib或者${FAISS_INSTALL_PATH}/lib64，与具体操作系统有关
        export FAISS_INSTALL_PATH_LIB=${FAISS_INSTALL_PATH}/lib
        mkdir -p ${FAISS_INSTALL_PATH} 
        sed -i "149 i virtual void search_with_filter (idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *mask = nullptr) const{}" Index.h     
        sed -i "49 i template <typename IndexT> IndexIDMapTemplate<IndexT>::IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids): index (index), own_fields (false) { this->is_trained = index->is_trained; this->metric_type = index->metric_type; this->verbose = index->verbose; this->d = index->d; id_map = ids; }" IndexIDMap.cpp     
        sed -i "30 i explicit IndexIDMapTemplate (IndexT *index, std::vector<idx_t> &ids);" IndexIDMap.h     
        sed -i "217 i utils/sorting.h" CMakeLists.txt   
        cd .. && cmake -B build . -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${FAISS_INSTALL_PATH}    
        make -C build -j faiss    
        make -C build -j swigfaiss    
        # 如果报错没有wheel，使用pip安装
        cd build/faiss/python && python3 setup.py bdist_wheel  
        cd ../../.. && make -C build install  
        cd build/faiss/python && cp libfaiss_python_callbacks.so ${FAISS_INSTALL_PATH_LIB}/ 
        cd dist
        # 该操作可能更新numpy到2.x.x版本，需要回退到1.26.4
        pip3 install faiss-1.10.0*.whl
        ```

    4.  按“Esc”键，输入<b>:wq!</b>，按“Enter”保存并退出编辑。
    5.  执行**install\_faiss.sh**脚本，安装faiss。

        ```
        bash install_faiss.sh
        ```

        > [!NOTE] 说明
        >-   如果报错没有wheel，请使用pip安装。
        >-   安装完faiss之后，numpy可能更新到2.x.x版本，需要回退至1.26.4。

6.  安装Index SDK。
    1.  增加对软件包的可执行权限。

        ```
        chmod +x Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run
        ```

    2.  执行如下命令，校验软件包的一致性和完整性。

        ```
        ./Ascend-mindxsdk-mxindex_{version}_linux-{arch}.run --check
        ```

        若显示如下信息，说明软件包已通过校验。

        ```
        Verifying archive integrity...  100%   SHA256 checksums are OK. All good.    
        ```

    3.  创建软件包的安装路径。
        -   **若用户未指定安装路径**，软件会默认安装到软件包所在的路径。
        -   **若用户想指定安装路径**，需要先创建安装路径。以安装路径“/home/HwHiAiUser/Ascend”为例：

            ```
            mkdir -p /home/HwHiAiUser/Ascend
            ```

    4.  安装Index SDK。

        ```
        ./Ascend-mindxsdk-mxindex_7.2.RC1_linux-aarch64.run --install --install-path=<安装路径> --platform=<npu_type>
        ```

        如需指定路径安装，请将<安装路径\>配置为上一步创建的路径，安装过程中提示Do you accept the EULA to install RAG SDK? \[Y/N\]时，输入Y或y，表示同意EULA协议，继续进行安装；输入其他字符时停止安装，退出程序。

        若出现如下回显信息，则表示软件成功安装。

        ```
        Uncompressing ASCEND MXINDEX RUN PACKAGE  100%
        ```

    5.  安装完Index SDK后执行Index SDK脚本。

        ```
        cd <安装路径>/mxIndex/ops && ./custom_opp_{arch}.run
        ```

7.  下载并安装ascendfaiss。
    1.  下载源码包并解压

        ```
        wget https://gitee.com/ascend/mindsdk-referenceapps/repository/archive/master.zip 
        unzip master.zip && cd mindsdk-referenceapps-master/IndexSDK/faiss-python
        ```

    2.  创建install\_ascendfaiss\_sh.sh脚本

        ```
        vi install_ascendfaiss.sh
        ```

    3.  在install\_ascendfaiss.sh脚本中写入如下内容。。

        ```       
        # 设置以下环境变量
        export PY_VERSION=python3.11
        export FAISS_INSTALL_PATH=/usr/local/faiss/faiss1.10.0
        # faiss安装后可能是${FAISS_INSTALL_PATH}/lib或者${FAISS_INSTALL_PATH}/lib64，与具体操作系统有关
        export FAISS_INSTALL_PATH_LIB=${FAISS_INSTALL_PATH}/lib
        export INDEXSDK_INSTALL_PATH=/home/HwHiAiUser/Ascend/mxIndex
        export PYTHON_HEADER=/usr/include/$PY_VERSION/
        export ASCEND_INSTALL_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
        export DRIVER_INSTALL_PATH=/usr/local/Ascend/
        export OPENBLAS_INSTALL_PATH=/home/HwHiAiUser/OpenBLAS
        export NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")
        swig -python -c++ -Doverride= -module swig_ascendfaiss -I${PYTHON_HEADER} -I${FAISS_INSTALL_PATH}/include -I${INDEXSDK_INSTALL_PATH}/include -DSWIGWORDSIZE64 -o swig_ascendfaiss.cpp swig_ascendfaiss.swig 
        
        g++ -std=c++11 -DFINTEGER=int -fopenmp -I/usr/local/include -I${ASCEND_INSTALL_PATH}/acllib/include -I${ASCEND_INSTALL_PATH}/runtime/include -fPIC -fstack-protector-all -Wall -Wreturn-type -D_FORTIFY_SOURCE=2 -g -O3 -Wall -Wextra -I${PYTHON_HEADER} -I${NUMPY_INCLUDE} -I${FAISS_INSTALL_PATH}/include -I${INDEXSDK_INSTALL_PATH}/include -c swig_ascendfaiss.cpp -o swig_ascendfaiss.o
        
        g++ -std=c++11 -shared -fopenmp -L${ASCEND_INSTALL_PATH}/lib64 -L${ASCEND_INSTALL_PATH}/acllib/lib64 -L${ASCEND_INSTALL_PATH}/runtime/lib64 -L${DRIVER_INSTALL_PATH}/driver/lib64 -L${DRIVER_INSTALL_PATH}/driver/lib64/common -L${DRIVER_INSTALL_PATH}/driver/lib64/driver -L${FAISS_INSTALL_PATH_LIB} -L${INDEXSDK_INSTALL_PATH}/lib -Wl,-rpath-link=${ASCEND_INSTALL_PATH}/acllib/lib64:${ASCEND_INSTALL_PATH}/runtime/lib64:${DRIVER_INSTALL_PATH}/driver/lib64:${DRIVER_INSTALL_PATH}/driver/lib64/common:${DRIVER_INSTALL_PATH}/driver/lib64/driver -L/usr/local/lib -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s -o _swig_ascendfaiss.so swig_ascendfaiss.o -L.. -lascendfaiss -lfaiss -lascend_hal -lc_sec 
        
        # 如果报错没有build，使用pip安装
        python3 -m build 
        # 该操作可能更新numpy到2.x.x版本，需要回退到1.26.4
        cd dist && pip3 install ascendfaiss*.whl
        export LD_LIBRARY_PATH=${INDEXSDK_INSTALL_PATH}/lib:${FAISS_INSTALL_PATH}/lib:$LD_LIBRARY_PATH
        ```

    4.  按“Esc”键，输入<b>:wq!</b>，按“Enter”保存并退出编辑。
    5.  执行**install\_ascendfaiss.sh**脚本，安装**ascendfaiss**。

        ```
        bash install_ascendfaiss.sh
        ```

8.  安装RAG SDK。

    ```
    bash  Ascend-mindxsdk-mxrag_<version>_linux-<arch>.run --install --install-path=<安装路径> --platform=<npu_type>
    # 安装第三方依赖包
    pip3 install  rank_bm25==0.2.2 langchain-opengauss==0.1.5
    # 安装依赖包
    pip3 install -r <安装路径>/mxRag/requirements.txt
    ```

    若出现如下回显信息，则表示软件成功安装。

    ```
    Install package successfully
    ```

    --install安装命令同时支持输入可选参数，如[表1](#table7138521890)所示。输入不在列表中的参数可能正常安装或者报错。

    > [!NOTICE] 须知
    >如果通过**./**_\{run\_file\_name\}__.run_** --help**命令查询出的参数未解释在如下表格，则说明该参数预留或适用于其他处理器版本，用户无需关注。

    **表 1**  安装包支持的参数表
    <a id="table7138521890"></a>
    <table><thead align="left"><tr id="row17138721693"><th class="cellrowborder" valign="top" width="30%" id="mcps1.2.3.1.1"><p id="p1413820210918"><a name="p1413820210918"></a><a name="p1413820210918"></a>输入参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="70%" id="mcps1.2.3.1.2"><p id="p31381421995"><a name="p31381421995"></a><a name="p31381421995"></a>含义</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row6139821299"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p16139921596"><a name="p16139921596"></a><a name="p16139921596"></a>--help | -h</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p191391221694"><a name="p191391221694"></a><a name="p191391221694"></a>查询帮助信息。</p>
    </td>
    </tr>
    <tr id="row185561236182712"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1771171662116"><a name="zh-cn_topic_0000001506619361_p1771171662116"></a><a name="zh-cn_topic_0000001506619361_p1771171662116"></a>--info</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p127115169219"><a name="zh-cn_topic_0000001506619361_p127115169219"></a><a name="zh-cn_topic_0000001506619361_p127115169219"></a>查询包构建信息。</p>
    </td>
    </tr>
    <tr id="row758753814276"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001506619361_p1571616102112"><a name="zh-cn_topic_0000001506619361_p1571616102112"></a><a name="zh-cn_topic_0000001506619361_p1571616102112"></a>--list</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001506619361_p2071101611219"><a name="zh-cn_topic_0000001506619361_p2071101611219"></a><a name="zh-cn_topic_0000001506619361_p2071101611219"></a>查询文件列表。</p>
    </td>
    </tr>
    <tr id="row15139102696"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1713913216912"><a name="p1713913216912"></a><a name="p1713913216912"></a>--check</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p10139124910"><a name="p10139124910"></a><a name="p10139124910"></a>查询包完整性。</p>
    </td>
    </tr>
    <tr id="row106251555135013"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1961118155117"><a name="p1961118155117"></a><a name="p1961118155117"></a>--quiet|-q</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p9611118135119"><a name="p9611118135119"></a><a name="p9611118135119"></a><span id="ph1426965871212"><a name="ph1426965871212"></a><a name="ph1426965871212"></a>表示静默操作，减少</span>人机交互的信息的打印。</p>
    </td>
    </tr>
    <tr id="row15381128165016"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p195393282506"><a name="p195393282506"></a><a name="p195393282506"></a>--nox11</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p153972811501"><a name="p153972811501"></a><a name="p153972811501"></a>废弃接口，无实际作用。</p>
    </td>
    </tr>
    <tr id="row87161623105113"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p12716112312514"><a name="p12716112312514"></a><a name="p12716112312514"></a>--noexec</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p169952029175415"><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p169952029175415"></a><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p169952029175415"></a>解压软件包到当前目录，但不执行安装脚本。配套--extract=&lt;path&gt;使用，格式为：--noexec --extract=&lt;path&gt;。</p>
    </td>
    </tr>
    <tr id="row199021147185119"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p090212473511"><a name="p090212473511"></a><a name="p090212473511"></a>--extract=&lt;path&gt;</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p109022478518"><a name="p109022478518"></a><a name="p109022478518"></a>解压软件包中文件到指定目录。可配套--noexec、--install、--upgrade之一参数使用。</p>
    </td>
    </tr>
    <tr id="row89711314205811"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1297217140583"><a name="p1297217140583"></a><a name="p1297217140583"></a>--tar arg1 [arg2 ...]</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p828614314480"><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p828614314480"></a><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p828614314480"></a>对软件包执行tar命令，使用tar后面的参数作为命令的参数。例如执行<strong id="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_b418314014491"><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_b418314014491"></a><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_b418314014491"></a>--tar xvf</strong>命令，解压run安装包的内容到当前目录。</p>
    </td>
    </tr>
    <tr id="row193861197593"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p8498171414590"><a name="p8498171414590"></a><a name="p8498171414590"></a>--version</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p134981514115910"><a name="p134981514115910"></a><a name="p134981514115910"></a>查询安装包<span id="ph14498101465911"><a name="ph14498101465911"></a><a name="ph14498101465911"></a>RAG SDK</span>版本。</p>
    </td>
    </tr>
    <tr id="row51391722098"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p191391721690"><a name="p191391721690"></a><a name="p191391721690"></a>--install</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p141392028910"><a name="p141392028910"></a><a name="p141392028910"></a>软件包安装操作命令。</p>
    </td>
    </tr>
    <tr id="row191392023915"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p613914210915"><a name="p613914210915"></a><a name="p613914210915"></a>--install-path=<em id="i6139127919"><a name="i6139127919"></a><a name="i6139127919"></a>&lt;path&gt;</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p41399211916"><a name="p41399211916"></a><a name="p41399211916"></a>（可选）自定义<span id="ph1058913412462"><a name="ph1058913412462"></a><a name="ph1058913412462"></a>RAG SDK</span>软件包安装根目录。如未设置，默认为当前命令执行所在目录。配置的路径必须是"/"或"~"开头，路径取值合法字符为"-_.0-9a-zA-Z/"，且不能包含“..，长度不超过1024。</p>
    <p id="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p1290010243505"><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p1290010243505"></a><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_p1290010243505"></a>若不指定，将安装到默认路径下：</p>
    <a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_ul1190072465010"></a><a name="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_ul1190072465010"></a><ul id="zh-cn_topic_0000001127124040_zh-cn_topic_0000001110810470_zh-cn_topic_0000001079598564_zh-cn_topic_0245337208_ul1190072465010"><li>若使用root用户安装，默认安装路径为：/usr/local/Ascend。</li><li>若使用非root用户安装，则默认安装路径为：${HOME}/Ascend。</li></ul>
    <p id="p86031014062"><a name="p86031014062"></a><a name="p86031014062"></a><span id="ph3593356171011"><a name="ph3593356171011"></a><a name="ph3593356171011"></a>若通过该参数指定了安装目录</span><span id="ph29251359141014"><a name="ph29251359141014"></a><a name="ph29251359141014"></a>，</span>该目录other用户不能有写权限，如果指定普通用户安装，安装目录属主必须为当前安装用户。</p>
    </td>
    </tr>
    <tr id="row151401321094"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p1514020212919"><a name="p1514020212919"></a><a name="p1514020212919"></a>--upgrade</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p71401326913"><a name="p71401326913"></a><a name="p71401326913"></a>软件包升级操作命令，将<span id="ph287118364464"><a name="ph287118364464"></a><a name="ph287118364464"></a>RAG SDK</span>升级到安装包所包含的版本。</p>
    </td>
    </tr>
    <tr id="row6140122998"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p111401321195"><a name="p111401321195"></a><a name="p111401321195"></a>--platform</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p121401723912"><a name="p121401723912"></a><a name="p121401723912"></a>可选参数，对应<span id="ph814022995"><a name="ph814022995"></a><a name="ph814022995"></a>昇腾AI处理器</span>类型。</p>
    <p id="p165441040184917"><a name="p165441040184917"></a><a name="p165441040184917"></a>请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。</p>
    <p id="p1541314212612"><a name="p1541314212612"></a><a name="p1541314212612"></a>若是<span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 超节点服务器</span>则取值为A3。</p>
    </td>
    </tr>
    <tr id="row138044192815"><td class="cellrowborder" valign="top" width="30%" headers="mcps1.2.3.1.1 "><p id="p3380154113280"><a name="p3380154113280"></a><a name="p3380154113280"></a>--whitelist</p>
    </td>
    <td class="cellrowborder" valign="top" width="70%" headers="mcps1.2.3.1.2 "><p id="p4380141182816"><a name="p4380141182816"></a><a name="p4380141182816"></a>可选参数，表示安装白名单特性模块，取值可以是operator或者whl，安装多个特性时，可以用逗号分隔，如果不设置该参数，则表示安装所有特性模块。</p>
    <p id="p13115204610243"><a name="p13115204610243"></a><a name="p13115204610243"></a>operator：指安装推理加速算子模块。</p>
    <p id="p4734513152515"><a name="p4734513152515"></a><a name="p4734513152515"></a>whl：指安装<span id="ph687723143313"><a name="ph687723143313"></a><a name="ph687723143313"></a>RAG SDK</span>功能模块（知识库管理、向量化、缓存等）。</p>
    </td>
    </tr>
    </tbody>
    </table>


    > [!NOTE] 说明
    >以下参数未展示在--help参数中，用户请勿直接使用。
    >-   --xwin：使用xwin模式运行。
    >-   --phase2：要求执行第二步动作。

9.  设置RAG SDK运行环境变量。
    1.  用vim打开文件\~/.bashrc，在文件最后添加如下内容。

        ```
        export MX_INDEX_FINALIZE=0
        export PY_VERSION=python3.11
        export LOGURU_FORMAT='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message!r}</level>'
        export MX_INDEX_MODELPATH=/home/HwHiAiUser/Ascend/modelpath
        # 设置index SDK安装路径，如果安装时未使用默认路径安装，请根据实际路径修改
        export MX_INDEX_INSTALL_PATH=/home/HwHiAiUser/Ascend/mxIndex
        export MX_INDEX_MULTITHREAD=1
        export ASCEND_HOME=$HOME/Ascend/
        export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/mxIndex/lib:/home/HwHiAiUser/faiss/faiss1.10.0/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=/home/HwHiAiUser/.local/lib/$PY_VERSION/site-packages/mx_rag/libs:$PYTHONPATH
        export LD_PRELOAD=$(ls /home/HwHiAiUser/.local/lib/$PY_VERSION/site-packages/scikit_learn.libs/libgomp-*):$LD_PRELOAD
        source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
        source /home/HwHiAiUser/Ascend/nnal/atb/set_env.sh
        source /home/HwHiAiUser/Ascend/mxRag/script/set_env.sh
        ```

    2.  保存退出后运行如下命令让环境生效。

        ```
        source ~/.bashrc
        ```

> [!NOTE] 说明 
>安装RAG SDK时可能出现报错信息：
>ERROR: Cannot uninstall 'xxx'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
>则说明xxx模块是操作系统自带的组件，无法直接升级，可以尝试重新下发指令安装：**pip3 install -r requirements.txt --ignore-installed**

**检查运行环境<a name="section19468866105"></a>**

1.  切换到运行用户HwHiAiUser。
2.  执行**npu-smi info**命令检查驱动是否挂载正常。

    如当Health参数的值为OK时，即表示当前芯片的健康状态为正常（以下仅为示例，请以实际查询到的信息为准）。

    ```
    +--------------------------------------------------------------------------------------------------------+
    | npu-smi 24.1.rc2                            Version: 24.1.rc2                                          |
    +-------------------------------+-----------------+------------------------------------------------------+
    | NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
    | Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
    +===============================+=================+======================================================+
    | 7       xxx                 | OK              | NA           44                0     / 0             |
    | 0       0                     | 0000:83:00.0    | 0            1851 / 21527                            |
    +===============================+=================+======================================================+
    | 8       xxx                 | OK              | NA           44                0     / 0             |
    | 0       1                     | 0000:84:00.0    | 0            1852 / 21527                            |
    +===============================+=================+======================================================+
    +-------------------------------+-----------------+------------------------------------------------------+
    | NPU     Chip                  | Process id      | Process name             | Process memory(MB)        |
    +===============================+=================+======================================================+
    | No running processes found in NPU 7                                                                    |
    +===============================+=================+======================================================+
    | No running processes found in NPU 8                                                                    |
    +===============================+=================+======================================================+
    ```


# 升级<a name="ZH-CN_TOPIC_0000001983329754"></a>

**注意事项<a name="section1894903161912"></a>**

升级操作涉及对安装目录的卸载再安装，如目录下存在其他文件，也会被一并删除。请在执行升级操作前，确保所有数据都已妥善处理。

**操作步骤<a name="section37391535123710"></a>**

用户如需将当前版本的RAG SDK升级至最新版本，可将最新的RAG SDK软件包上传至安装环境后，在软件包所在目录下使用命令进行版本升级，具体命令参见如下。以下命令运行用户为HwHiAiUser。

1.  使用<b>--upgrade</b>命令升级。

    ```
    bash Ascend-mindxsdk-mxrag_<version>_linux-<arch>.run --upgrade --install-path=<安装路径> --platform=<npu_type>
    ```

    <i><version\></i>为版本号，<i><arch\></i>为操作系统架构，<i><npu\_type\></i>为芯片类型。

    **表 1**  参数名及说明

    <a name="table17754104316374"></a>
    <table><thead align="left"><tr id="row575494393716"><th class="cellrowborder" valign="top" width="35.18%" id="mcps1.2.3.1.1"><p id="p1875474393717"><a name="p1875474393717"></a><a name="p1875474393717"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="64.82%" id="mcps1.2.3.1.2"><p id="p375584303712"><a name="p375584303712"></a><a name="p375584303712"></a>参数说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1975564333719"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p37557431375"><a name="p37557431375"></a><a name="p37557431375"></a>--upgrade</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p77551431377"><a name="p77551431377"></a><a name="p77551431377"></a>软件包升级操作命令，将<span id="ph1656424955418"><a name="ph1656424955418"></a><a name="ph1656424955418"></a>RAG SDK</span>升级到安装包所包含的版本。</p>
    </td>
    </tr>
    <tr id="row167552043133716"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3755943183713"><a name="p3755943183713"></a><a name="p3755943183713"></a>--platform</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p0755943163719"><a name="p0755943163719"></a><a name="p0755943163719"></a>对应<span id="ph075594383717"><a name="ph075594383717"></a><a name="ph075594383717"></a>昇腾AI处理器</span>类型。</p>
    <p id="p1399417513288"><a name="p1399417513288"></a><a name="p1399417513288"></a>请在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”最后一位数字删掉，即是--platform的取值。</p>
    <p id="p145658372256"><a name="p145658372256"></a><a name="p145658372256"></a>若是<span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 超节点服务器</span>则取值为A3。</p>
    </td>
    </tr>
    <tr id="row15756174312379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p12756114363719"><a name="p12756114363719"></a><a name="p12756114363719"></a>--install-path</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p19756943133712"><a name="p19756943133712"></a><a name="p19756943133712"></a>（可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。配置的路径必须是"/"或"~"开头，路径取值合法字符为"a-zA-Z0-9_/-"。</p>
    <p id="p14756174303720"><a name="p14756174303720"></a><a name="p14756174303720"></a>如使用自定义目录安装，建议在升级操作时使用该参数。</p>
    <p id="p1637441211"><a name="p1637441211"></a><a name="p1637441211"></a>请确保配置的路径下已安装<span id="ph431844192119"><a name="ph431844192119"></a><a name="ph431844192119"></a>RAG SDK</span>。</p>
    </td>
    </tr>
    <tr id="row97569438379"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p1975694363720"><a name="p1975694363720"></a><a name="p1975694363720"></a>--quiet</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p1075634353718"><a name="p1075634353718"></a><a name="p1075634353718"></a><span id="ph264718181219"><a name="ph264718181219"></a><a name="ph264718181219"></a>表示静默操作</span><span id="ph1127692317123"><a name="ph1127692317123"></a><a name="ph1127692317123"></a>。</span></p>
    </td>
    </tr>
    <tr id="row38771538191618"><td class="cellrowborder" valign="top" width="35.18%" headers="mcps1.2.3.1.1 "><p id="p3380154113280"><a name="p3380154113280"></a><a name="p3380154113280"></a>--whitelist</p>
    </td>
    <td class="cellrowborder" valign="top" width="64.82%" headers="mcps1.2.3.1.2 "><p id="p4380141182816"><a name="p4380141182816"></a><a name="p4380141182816"></a>可选参数，表示安装白名单特性，取值可以是operator或者whl，安装多个特性时，可以用逗号分隔。</p>
    </td>
    </tr>
    </tbody>
    </table>

    升级过程中提示Do you want to upgrade to a newer version provided by this package and the old version will be removed? \[Y/N\]时，输入Y或y表示同意升级，此时旧版本的RAG SDK将被卸载；输入其他内容表示退出升级。

2.  出现以下提示说明升级成功。

    ```
    Upgrade RAG SDK successfully
    ```

# 卸载<a name="ZH-CN_TOPIC_0000002018595337"></a>

用户如需移除RAG SDK软件包部署，可参考以下命令进行卸载。使用HwHiAiUser用户执行：

```
bash 安装目录/mxRag/script/uninstall.sh
```

若显示如下信息，则表示软件成功卸载。

```
Uninstall RAG SDK package successfully.
```



