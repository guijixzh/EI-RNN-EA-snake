# Third-Party Notices

本仓库包含或引用了以下第三方开源项目。第三方组件的版权归原作者所有，
其许可条款仅适用于相应组件。

## chynl/snake

- 来源：<https://github.com/chynl/snake>（原 `chuyangliu/snake`）
- 许可证：MIT License，完整文本见 [`third_party/chynl/LICENSE`](third_party/chynl/LICENSE)
- 仓库内副本：[`third_party/chynl/`](third_party/chynl/)（在各源文件头部注明了来源与
  SPDX 标识，其余内容保持上游原样）
- 派生使用：[`experiments/test7_series/ref_solver.py`](experiments/test7_series/ref_solver.py)
  为其图搜索解法器（GraphAgent / CycleSolver）的移植版本。

## Ackeraa/snake

- 来源：<https://github.com/Ackeraa/snake>
- 许可证：**上游未声明任何许可证**（默认保留所有权利，all rights reserved）。
- 因此，其代码与预训练权重（`nn.py`、`nn_97.pth` 等）**不随本仓库分发**，仅在文档
  中作为外部对照基准引用。如需运行相关脚本，请自行从上游获取并放入
  `third_party/ackeraa/`（该目录被 `.gitignore` 忽略），或通过环境变量
  `ACKERAA_DIR` 指向本地目录。
