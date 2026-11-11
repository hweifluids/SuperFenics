# Centralize already-cloned official demos without changing their source files.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$workspacePrefix = $workspaceRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

function Get-WorkspacePath([string]$RelativePath) {
    $fullPath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $RelativePath))
    if (-not $fullPath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside this workspace: $fullPath"
    }
    return $fullPath
}

function Get-FileInventory([string]$Directory) {
    $prefix = $Directory.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $entries = Get-ChildItem -LiteralPath $Directory -Force -Recurse
    if (@($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -ne 0) {
        throw "Unexpected link inside demo source: $Directory"
    }
    return @($entries | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        [pscustomobject]@{
            path = $_.FullName.Substring($prefix.Length).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    } | Sort-Object path)
}

function New-CompatibilityLink([string]$LinkPath, [string]$TargetPath) {
    if (Test-Path -LiteralPath $LinkPath) {
        $item = Get-Item -LiteralPath $LinkPath -Force
        if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Compatibility path already contains ordinary files: $LinkPath"
        }
        $actualTarget = [IO.Path]::GetFullPath([string]$item.Target)
        if ($actualTarget -ne [IO.Path]::GetFullPath($TargetPath)) {
            throw "Compatibility link points elsewhere: $LinkPath -> $actualTarget"
        }
        return
    }
    $parent = Split-Path -Parent $LinkPath
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $linkType = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
    New-Item -ItemType $linkType -Path $LinkPath -Target $TargetPath | Out-Null
}

$mappings = @(
    @{ module = 'dolfinx'; source = 'dolfinx/python/demo'; destination = 'examples/dolfinx/python/demo' }
    @{ module = 'dolfinx'; source = 'dolfinx/cpp/demo'; destination = 'examples/dolfinx/cpp/demo' }
    @{ module = 'basix'; source = 'basix/demo'; destination = 'examples/basix/demo' }
    @{ module = 'ufl'; source = 'ufl/demo'; destination = 'examples/ufl/demo' }
    @{ module = 'ffcx'; source = 'ffcx/demo'; destination = 'examples/ffcx/demo' }
)
$records = @()
foreach ($mapping in $mappings) {
    $source = Get-WorkspacePath $mapping.source
    $destination = Get-WorkspacePath $mapping.destination
    if (Test-Path -LiteralPath $source) {
        $sourceItem = Get-Item -LiteralPath $source -Force
        if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            New-CompatibilityLink $source $destination
        } else {
            $before = Get-FileInventory $source
            if (Test-Path -LiteralPath $destination) {
                $after = Get-FileInventory $destination
                if (($before | ConvertTo-Json -Compress) -ne ($after | ConvertTo-Json -Compress)) {
                    throw "Both demo locations exist and differ; preserve both for review: $source and $destination"
                }
                # A fresh recursive clone contains both identical Git snapshots.
                # Preserve the original directory in ignored local storage before linking.
                $backupName = $mapping.source.Replace('/', '-') + '-' + [guid]::NewGuid().ToString('N')
                $backup = Get-WorkspacePath ('.tests/workspace-layout/backups/' + $backupName)
                New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
                Write-Output "ARCHIVE identical duplicate $source -> $backup"
                Move-Item -LiteralPath $source -Destination $backup
            } else {
                New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
                Write-Output "MOVE $source -> $destination"
                Move-Item -LiteralPath $source -Destination $destination
                $after = Get-FileInventory $destination
                if (($before | ConvertTo-Json -Compress) -ne ($after | ConvertTo-Json -Compress)) {
                    throw "Moved files differ; preserve the destination for review: $destination"
                }
            }
            New-CompatibilityLink $source $destination
        }
    } elseif (Test-Path -LiteralPath $destination) {
        New-CompatibilityLink $source $destination
    } else {
        throw "Official demo source is missing; initialize the Git submodules first: $source"
    }
    $inventory = Get-FileInventory $destination
    $records += [pscustomobject]@{
        module = $mapping.module
        source = $mapping.source
        destination = $mapping.destination
        files = $inventory
    }
    Write-Output "VERIFIED $($mapping.destination): $($inventory.Count) files"
}

# Preserve relative include paths in the unmodified upstream demo helpers.
New-CompatibilityLink (Get-WorkspacePath 'examples/basix/cmake') (Get-WorkspacePath 'basix/cmake')
New-CompatibilityLink (Get-WorkspacePath 'examples/ffcx/ffcx') (Get-WorkspacePath 'ffcx/ffcx')

$localOutput = Get-WorkspacePath '.tests/workspace-layout'
New-Item -ItemType Directory -Path $localOutput -Force | Out-Null
$records | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $localOutput 'layout-inventory.json') -Encoding UTF8
Write-Output 'Official examples are centralized; original demo paths remain compatible.'
