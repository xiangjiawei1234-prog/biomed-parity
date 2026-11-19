# 发布流程

本项目通过 PyPI 发布，推荐使用 GitHub Actions 与 PyPI Trusted Publishing（OIDC），无需在仓库中保存 API token。首次发布前需完成一次性配置，之后每个版本由 GitHub Release 或手动触发的 `Publish to PyPI` 工作流上传。

## 一次性配置

1. 注册 PyPI 账号并启用双因素认证（PyPI 强制要求）。
2. 配置 Trusted Publisher（推荐）：
   - 登录 PyPI，进入 Account settings → Publishing → Add a new pending publisher；
   - PyPI Project Name：`biomed-parity`
   - Owner：`xiangjiawei1234-prog`
   - Repository name：`biomed-parity`
   - Workflow name：`publish.yml`
   - Environment name：`pypi`
3. 使用 API token 发布时，在 PyPI 创建项目或账号级 token，通过环境变量 `UV_PUBLISH_TOKEN` 传给 `uv publish`。token 不得写入仓库、脚本或提交历史。

## 版本发布

1. 修改版本号：编辑 `src/biomed_parity/__init__.py` 中的 `__version__`。`pyproject.toml` 通过 `tool.setuptools.dynamic` 读取该值。
2. 提交并推送，确认 CI 通过。
3. 在 GitHub 创建 Release，tag 与版本号对应（例如 `v0.2.0`），触发 `Publish to PyPI` 工作流。
4. 也可在 Actions 页面手动运行 `Publish to PyPI`；或在本地执行：

   ```sh
   uv build
   uvx twine check dist/*
   UV_PUBLISH_TOKEN=<token> uv publish
   ```

5. 发布完成后检查 https://pypi.org/project/biomed-parity/ 与 `pip install biomed-parity`。

## 注意事项

- PyPI 的版本号与上传文件不可覆盖。发布后需要修改元数据或文档时，应提升版本号并发布新版本。
- Trusted Publishing 仅接受配置中指定的 owner、仓库、工作流与 environment 发起的上传。
- 首次发布版本为 `0.1.0`。
