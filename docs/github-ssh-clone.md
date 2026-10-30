# 私有仓库 SSH 克隆：命令记录

日期：2026-10-30。适用位置：Linux 服务器终端，原操作目录为 `/data/X-D-Lab/ty`。

背景：HTTPS 克隆 `ccathead/sentiment-report` 时出现 `gnutls_handshake() failed: The TLS connection was non-properly terminated`。该报错本身不能证明仓库权限有问题；这里改用 SSH 连接。服务器上的 SSH 认证、端口连通性与实际克隆结果均**待核验**。

来源：[原对话：解决私有仓库克隆](chatgpt-conversation://6ae449f0-e324-83ec-86fd-d636b9e80ae5)。读取时助手正文仅返回引用占位符，以下为依据 GitHub 官方文档补全的操作记录，不是原回答逐字转录。

## 1. 在服务器生成密钥

以下命令在服务器上逐条输入。不要复制终端提示符 `c1-505@...$`；代码块中的命令可以直接复制。

```bash
cd /data/X-D-Lab/ty
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ls -al ~/.ssh
```

先检查有无可复用的密钥。下面使用专用文件名；如果已经存在 `id_ed25519_github`，不要覆盖，确认后复用或另起文件名，并同步修改后续命令。

```bash
ssh-keygen -t ed25519 -C "github-sentiment-report" -f ~/.ssh/id_ed25519_github
```

交互输入说明：

- `Enter passphrase (empty for no passphrase):`：输入自设的密钥口令，输入时屏幕不显示字符；直接回车表示不设口令。
- `Enter same passphrase again:`：再次输入同一口令。
- 如果提示 `Overwrite (y/n)?`：输入 `n`，避免覆盖已有密钥。

启动密钥代理并加载密钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub
```

若提示 `Enter passphrase for ...`，输入刚设置的密钥口令。

## 2. 在 GitHub 网页添加公钥

复制上一条 `cat` 输出的整行公钥（以 `ssh-ed25519` 开头），在有该私有仓库访问权限的 GitHub 账号中打开 **Settings → SSH and GPG keys → New SSH key**：Title 可填 `c1505-server`，Key type 选择 `Authentication Key`，将公钥粘贴到 Key 后保存。

只复制 `.pub` 文件内容；不复制或上传不带 `.pub` 的私钥。SSH key 关联账号后，该账号仍需拥有仓库权限。

## 3. 测试并克隆

```bash
ssh -T -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes git@github.com
```

首次出现 `Are you sure you want to continue connecting ...?` 时，与 [GitHub 公布的主机指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)核对，一致后输入 `yes` 并回车。

出现 `Hi 用户名! You've successfully authenticated, but GitHub does not provide shell access.` 表示认证成功；核对用户名是否为有仓库权限的账号。这个测试即使成功也可能返回退出码 `1`。

在当前终端设置本次会话的 Git SSH 参数，再克隆：

```bash
export GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes'
cd /data/X-D-Lab/ty
git clone git@github.com:ccathead/sentiment-report.git
cd sentiment-report
git remote -v
```

本地笔记目录叫 `sentiment_report`，服务器克隆后的默认目录叫 `sentiment-report`。如果目标目录已存在，先检查内容，不删除或覆盖；若它已是完整仓库，可直接使用下一节的切换命令。

`export` 仅作用于当前终端；新开终端时按需重新执行 `eval`、`ssh-add` 和 `export`。

## 4. 22 端口不通时，改用 SSH 的 443 端口

如果第 3 节测试超时或连接被拒绝，先测试备用地址：

```bash
ssh -T -p 443 -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes git@ssh.github.com
```

同样核对主机指纹；认证成功后执行：

```bash
export GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes'
cd /data/X-D-Lab/ty
git clone ssh://git@ssh.github.com:443/ccathead/sentiment-report.git
cd sentiment-report
git remote -v
```

443 端口使用的主机名是 `ssh.github.com`。此方法仍使用 SSH，网络或代理限制仍可能影响连接。

如果**已经克隆成功**，只需切换已有仓库的远端地址，按网络情况二选一：

```bash
cd /data/X-D-Lab/ty/sentiment-report
# 常规 SSH（22 端口）
git remote set-url origin git@github.com:ccathead/sentiment-report.git
# 或使用备用 443 端口（选此项时执行下面这条代替上一条）
git remote set-url origin ssh://git@ssh.github.com:443/ccathead/sentiment-report.git
git remote -v
```

## 常见提示

| 提示 | 下一步 |
|---|---|
| `Permission denied (publickey)` | 检查公钥是否添加到了正确账号，测试是否指定了对应私钥 |
| `Repository not found` | 检查仓库地址、测试显示的 GitHub 用户名及该账号的私有仓库权限 |
| `Connection timed out` / `Connection refused` | 尝试第 4 节；443 也不通时，网络原因待核验 |
| `destination path ... already exists` | 检查已有目录；完整仓库可直接进入并切换远端，不重复克隆 |

克隆只获取代码；冻结数据包仍需单独传入，后续见[实验运行指南](experiment-guide.md)。

## 官方依据

- [生成 SSH 密钥与添加至 ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [添加公钥到 GitHub 账号](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- [测试 SSH 连接](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)
- [通过 443 端口使用 SSH](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port)
