$log = Join-Path $PSScriptRoot "_nat_log.txt"
"=== NAT setup started $(Get-Date) ===" | Out-File $log

foreach ($svc in @("winnat","ipnat")) {
    try {
        Set-Service -Name $svc -StartupType Automatic -ErrorAction SilentlyContinue
        Start-Service -Name $svc -ErrorAction Stop
        "$svc started" | Out-File $log -Append
    } catch { "$svc start ERR: $_" | Out-File $log -Append }
}
Start-Sleep -Seconds 2

try {
    Set-NetIPInterface -InterfaceAlias "Ethernet" -Forwarding Enabled -ErrorAction Stop
    Set-NetIPInterface -InterfaceAlias "Wi-Fi" -Forwarding Enabled -ErrorAction Stop
    "forwarding enabled Ethernet+WiFi" | Out-File $log -Append
} catch { "forwarding ERR: $_" | Out-File $log -Append }

try {
    Get-NetNat -Name "AgibotNat" -ErrorAction SilentlyContinue | Remove-NetNat -Confirm:$false -ErrorAction SilentlyContinue
} catch {}
try {
    New-NetNat -Name "AgibotNat" -InternalIPInterfaceAddressPrefix "10.0.1.0/24" -ErrorAction Stop | Out-Null
    "New-NetNat AgibotNat 10.0.1.0/24: CREATED" | Out-File $log -Append
} catch { "New-NetNat ERR: $_" | Out-File $log -Append }

"=== Get-NetNat ===" | Out-File $log -Append
Get-NetNat -ErrorAction SilentlyContinue | Format-List Name,InternalIPInterfaceAddressPrefix,Active | Out-String | Out-File $log -Append
"=== services now ===" | Out-File $log -Append
Get-Service winnat,ipnat | Select-Object Name,Status | Format-Table -AutoSize | Out-String | Out-File $log -Append
"=== DONE ===" | Out-File $log -Append
