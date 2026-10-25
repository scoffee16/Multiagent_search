<#
.SYNOPSIS
    为 MultiAgent-Search 准备一份与 ragflow-sdk==0.23.1 对齐的 RAGFlow v0.23.1 部署副本。

.DESCRIPTION
    做三件事：
      1. 在 D:\RAGflow\ragflow-v0.23.1 检出 tag v0.23.1（用 git worktree，不动你现有 main 检出）
      2. 改写该副本 docker/.env 的宿主端口，避开本机已占用的 80/443/9000/9001/5432
      3. 打印后续启动命令与配置清单

    需要写成宿主端口的原因：com.docker.backend 已占用 80/443/9000/9001/5432（本机在跑 Dify 那套容器），
    直接沿用默认端口会 "port is already allocated"。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_ragflow_v0231.ps1
    只准备，不启动。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_ragflow_v0231.ps1 -Up
    准备 + 立刻 docker compose up -d（会拉约 9GB 镜像，前提是 Docker 代理已配好）。
#>
[CmdletBinding()]
param(
    # 准备完直接 docker compose up -d
    [switch]$Up,

    # 文档引擎：elasticsearch（官方默认，稳）或 infinity（更省内存，ES 起不来时的替代）
    [ValidateSet('elasticsearch', 'infinity')]
    [string]$Engine = 'elasticsearch',

    # 用 slim 镜像（约 2GB，省流量；但要在 RAGFlow 网页里自己配 embedding/rerank 模型）
    [switch]$Slim
)

$ErrorActionPreference = 'Stop'

$repo   = 'D:\RAGflow\ragflow'
$target = 'D:\RAGflow\ragflow-v0.23.1'
$tag    = 'v0.23.1'

function Write-Step($n, $msg) { Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Warn2($msg)   { Write-Host "    !!  $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 0. 前置检查
Write-Step '0/3' '前置检查'

if (-not (Test-Path $repo)) { throw "找不到 RAGFlow 仓库：$repo" }
$gitDir = Join-Path $repo '.git'
if (-not (Test-Path $gitDir)) { throw "$repo 不是 git 仓库（没有 .git）" }
Write-Ok "仓库存在：$repo"

$commit = (& git -C $repo rev-parse --verify "$tag^{commit}" 2>$null)
if (-not $commit) {
    throw "仓库里没有 tag $tag。先在 $repo 里执行: git fetch --tags"
}
Write-Ok "tag $tag -> $commit"

# 版本自检：要部署的 RAGFlow 版本必须和项目 requirements.txt 里的 ragflow-sdk 对齐
$reqFile = Join-Path $PSScriptRoot '..\requirements.txt'
if (Test-Path $reqFile) {
    $sdk = (Select-String -Path $reqFile -Pattern '^\s*ragflow-sdk\s*==\s*([\w\.\-]+)' |
            Select-Object -First 1).Matches.Groups[1].Value
    if ($sdk) {
        if ($tag.TrimStart('v') -ne $sdk) {
            Write-Warn2 "requirements.txt 里是 ragflow-sdk==$sdk，而这里部署 $tag，两者不一致会导致接口报错"
        } else {
            Write-Ok "版本对齐：ragflow-sdk==$sdk <-> RAGFlow $tag"
        }
    }
}

try {
    $composeVer = (& docker compose version 2>&1 | Select-Object -First 1)
    Write-Ok "docker compose: $composeVer"
    if ($composeVer -match 'v2\.(\d+)\.' -and [int]$Matches[1] -lt 20) {
        Write-Warn2 'compose 版本低于 v2.20，docker-compose.yml 里的 include: 可能不支持，建议升级 Docker Desktop'
    }
} catch {
    Write-Warn2 '没检测到 docker compose（Docker Desktop 开着吗？）'
}

# ---------------------------------------------------------------- 1. 工作副本
Write-Step '1/3' "准备 v0.23.1 工作副本：$target"

if (Test-Path (Join-Path $target 'docker\.env')) {
    Write-Ok '工作副本已存在，跳过检出'
} elseif (Test-Path $target) {
    throw "$target 已存在但不是完整的 v0.23.1 工作副本，请先手动删除它再重跑本脚本"
} else {
    & git -C $repo worktree add $target $tag
    if ($LASTEXITCODE -ne 0) { throw "git worktree add 失败（退出码 $LASTEXITCODE）" }
    if (-not (Test-Path (Join-Path $target 'docker\.env'))) { throw '检出后仍找不到 docker/.env' }
    Write-Ok "已检出到 $target（你的 main 检出 D:\RAGflow\ragflow 不受影响）"
}

# ---------------------------------------------------------------- 2. 改端口
Write-Step '2/3' '改写 docker/.env：端口 + 文档引擎'

$envFile = Join-Path $target 'docker\.env'

# 宿主端口全部避开 Dify：80->8088, 443->8443, 9000->9010, 9001->9011, 5432->5433
$settings = [ordered]@{
    'SVR_WEB_HTTP_PORT'  = '8088'   # RAGFlow 网页 / 登录页
    'SVR_WEB_HTTPS_PORT' = '8443'
    'MINIO_PORT'         = '9010'
    'MINIO_CONSOLE_PORT' = '9011'
    'DOC_ENGINE'         = $Engine
}
if ($Engine -eq 'infinity') {
    $settings['INFINITY_PSQL_PORT'] = '5433'   # 本机 5432 被 Dify 的 postgres 占用
}

$raw = [System.IO.File]::ReadAllText($envFile)
foreach ($k in $settings.Keys) {
    $v   = $settings[$k]
    $pat = '(?m)^[ \t]*' + [regex]::Escape($k) + '[ \t]*=.*$'
    if ([regex]::IsMatch($raw, $pat)) {
        $raw = [regex]::Replace($raw, $pat, "$k=$v")
    } else {
        if (-not $raw.EndsWith("`n")) { $raw += "`n" }
        $raw += "$k=$v`n"
    }
}

if ($Slim) {
    $raw = [regex]::Replace($raw, '(?m)^[ \t]*RAGFLOW_IMAGE[ \t]*=.*$', 'RAGFLOW_IMAGE=infiniflow/ragflow:v0.23.1-slim')
}

[System.IO.File]::WriteAllText($envFile, $raw, (New-Object System.Text.UTF8Encoding($false)))
Write-Ok "wrote $envFile"

Write-Host "    端口映射（宿主）：" -ForegroundColor DarkGray
Write-Host "      网页/登录  http://127.0.0.1:8088      : 80"    -ForegroundColor DarkGray
Write-Host "      后端 API   http://127.0.0.1:9380      : 9380"  -ForegroundColor DarkGray
Write-Host "      MySQL      127.0.0.1:5455  -> 3306"          -ForegroundColor DarkGray
Write-Host "      Elasticsearch 127.0.0.1:1200 -> 9200  (engine=$Engine)" -ForegroundColor DarkGray

$imgLine = (Select-String -Path $envFile -Pattern '^\s*RAGFLOW_IMAGE\s*=' | Select-Object -First 1).Line
Write-Ok $imgLine

# ---------------------------------------------------------------- 3. 下一步
Write-Step '3/3' '下一步'

$composeDir = Join-Path $target 'docker'

if ($Up) {
    Write-Host "    正在启动（首次会拉约 9GB 镜像，慢是正常的）..." -ForegroundColor Yellow
    Push-Location $composeDir
    try {
        & docker compose -f docker-compose.yml up -d
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code -ne 0) {
        Write-Warn2 "docker compose 退出码 $code，看上面的报错"
    } else {
        Write-Ok '已拉起，用 docker compose ps 看健康状态'
    }
} else {
    Write-Host ''
    Write-Host '  手动启动（在下面的目录里执行）：' -ForegroundColor White
    Write-Host "    cd $composeDir" -ForegroundColor White
    Write-Host '    docker compose -f docker-compose.yml up -d' -ForegroundColor White
    Write-Host '    docker compose ps' -ForegroundColor White
}

Write-Host ''
Write-Host '  起来之后还要做（缺一步 RAGFlow 工具就会返回“没有任何可用助手”）：' -ForegroundColor White
Write-Host '    1. 浏览器打开 http://127.0.0.1:8088 ，注册/登录' -ForegroundColor White
Write-Host '    2. 右上角头像 -> API -> 复制 API Key（形如 ragflow-xxxxxxxx）' -ForegroundColor White
Write-Host '    3. 新建知识库 -> 上传文档 -> 等解析完成' -ForegroundColor White
Write-Host '    4. 新建【聊天助手 Chat】并关联该知识库（工具查的是 Chat，不是知识库）' -ForegroundColor White
Write-Host '    5. 把下面两行写进项目根目录 .env：' -ForegroundColor White
Write-Host '         RAGFLOW_API_URL=http://127.0.0.1:9380' -ForegroundColor White
Write-Host '         RAGFLOW_API_KEY=ragflow-xxxxxxxxxxxxxxxx' -ForegroundColor White
Write-Host ''
Write-Host '  如果 es01 起不来（日志里出现 vm.max_map_count）：' -ForegroundColor Yellow
Write-Host '    wsl -d docker-desktop sysctl -w vm.max_map_count=262144' -ForegroundColor Yellow
Write-Host '    或改用 ./scripts/setup_ragflow_v0231.ps1 -Engine infinity' -ForegroundColor Yellow
