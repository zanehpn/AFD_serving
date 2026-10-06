# AFD_serving

ECO / AFD 配置搜索与服务实验代码。来源为本机 `publication/MOE_DVFS`，
代码提交 `9193339ae91495d979d424bdc76d0775a5883f96`。

- `bo_dse/native.py`：搜索与部署入口。
- `bo_dse/scripts/afd/static_dse/`：配置空间、GP、BO/GA/Random、校准和配置冻结。
- `bo_dse/scripts/afd/four_stage_dse_v6/`：四阶段与 FIFO 模型。
- `migration/`、`scripts/`、`services/`：安装、回放、遥测和运行工具。
- `environment/`、`inputs/`：依赖锁定与运行配置。
- `patches/`、`runtime-patches/`：后端补丁。
- `tests/`、`bo_dse/tests/`：测试。

CPU 检查：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash bo_dse/test_cpu.sh
```

依赖版本见 `bo_dse/requirements-cpu.txt`。这里只保留代码和运行配置，
不包含论文、绘图数据或实验结果归档。原有 `analyze_m1_repetitions.py` 保留。

请求 trace 与历史 calibration 数据未打包；运行相应回放或旧迁移流程前需自行提供输入数据。

项目配置中的路径相对于仓库根目录；从仓库根目录运行命令。脚本内部可根据自身位置
计算运行路径，不依赖原机器目录。CUDA 工具使用当前 `PATH`；如需兼容库，显式设置
`ECODEP_CUDA_COMPATIBILITY_PATH`。操作系统接口、解释器 shebang 和容器内挂载路径
属于运行协议，不是宿主机的个人目录。

公开代码检查：`python3 tools/check_public_code.py`。输出只包含文件、行号和问题类型，
不打印敏感值；二进制文件和 Git 历史需另行检查。不要提交密钥、真实请求数据或运行日志。
自定义插件 bundle 的提交身份已匿名化，运行时代码保持一致，测试示例路径改为相对路径，锁定文件已更新；
其提交 ID 与历史实验不同。移植后的配置和源码校验值也已更新，不能作为原始实验冻结证据。
