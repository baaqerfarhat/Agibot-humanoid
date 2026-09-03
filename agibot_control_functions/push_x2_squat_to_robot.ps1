# Copy the 40% squat policy + deploy scripts onto the robot.
# Run from the Windows laptop on the robot Ethernet (10.0.1.50 -> 10.0.1.41).
#
#   .\agibot_control_functions\push_x2_squat_to_robot.ps1
param(
    [string]$RobotHost = "run@10.0.1.41",
    [string]$Key = "$env:USERPROFILE\.ssh\agibot_ed25519"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PolicyName = "x2_squat_policy_40pct_iter16499.npz"
$PolicySrc = Join-Path $Root "agibot_control_functions\policies\$PolicyName"

if (-not (Test-Path $PolicySrc)) { throw "missing $PolicySrc" }

$ssh = @()
if (Test-Path $Key) { $ssh = @("-i", $Key) }
$ssh += @("-o", "ConnectTimeout=5", $RobotHost)

function Invoke-Robot([string]$Cmd) {
    & ssh @ssh $Cmd
}

Write-Host "===== copy squat policy + scripts to $RobotHost ====="
Invoke-Robot "mkdir -p agibot_control_functions/policies"

$scpKey = @()
if (Test-Path $Key) { $scpKey = @("-i", $Key) }

& scp @scpKey $PolicySrc "${RobotHost}:agibot_control_functions/policies/$PolicyName"

$files = @(
    "deploy_x2_squat.py",
    "base_frame.py",
    "robot_states_control.py",
    "run_logger.py",
    "run_x2_squat.sh"
)
foreach ($f in $files) {
    & scp @scpKey (Join-Path $Root "agibot_control_functions\$f") "${RobotHost}:agibot_control_functions/$f"
}

Invoke-Robot "chmod +x agibot_control_functions/run_x2_squat.sh"
Invoke-Robot "ls -lh agibot_control_functions/policies/$PolicyName agibot_control_functions/run_x2_squat.sh"

Write-Host ""
Write-Host "On the robot:"
Write-Host "  ssh $($ssh -join ' ')"
Write-Host "  aima em stop-app mc          # on 10.0.1.40, before any --engage"
Write-Host "  cd ~/agibot_control_functions && ./run_x2_squat.sh"
Write-Host "  # first --engage: SUSPENDED"
Write-Host "  ./run_x2_squat.sh --engage"
