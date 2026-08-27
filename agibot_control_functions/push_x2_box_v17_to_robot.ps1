# Copy v17 iter 49000 box-walk policy + deploy scripts onto the robot.
# Run from the Windows laptop on the robot Ethernet (10.0.1.50 -> 10.0.1.41).
#
#   .\agibot_control_functions\push_x2_box_v17_to_robot.ps1
param(
    [string]$RobotHost = "run@10.0.1.41",
    [string]$Key = "$env:USERPROFILE\.ssh\agibot_ed25519"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PolicyName = "x2_box_policy_walk_feasible_v17_iter49000.npz"
$PolicySrc = Join-Path $Root "box_pickup\policy\$PolicyName"

if (-not (Test-Path $PolicySrc)) { throw "missing $PolicySrc" }

$ssh = @()
if (Test-Path $Key) { $ssh = @("-i", $Key) }
$ssh += @("-o", "ConnectTimeout=5", $RobotHost)

function Invoke-Robot([string]$Cmd) {
    & ssh @ssh $Cmd
}

Write-Host "===== copy policy + scripts to $RobotHost ====="
Invoke-Robot "mkdir -p agibot_control_functions/policies box_pickup/policy"

$scpKey = @()
if (Test-Path $Key) { $scpKey = @("-i", $Key) }

& scp @scpKey $PolicySrc "${RobotHost}:agibot_control_functions/policies/$PolicyName"
& scp @scpKey $PolicySrc "${RobotHost}:box_pickup/policy/$PolicyName"

$files = @(
    "deploy_x2_box_pickup.py",
    "base_frame.py",
    "robot_states_control.py",
    "run_logger.py",
    "run_x2_box_v17.sh",
    "_dryrun_box.sh"
)
foreach ($f in $files) {
    & scp @scpKey (Join-Path $Root "agibot_control_functions\$f") "${RobotHost}:agibot_control_functions/$f"
}

Invoke-Robot "chmod +x agibot_control_functions/run_x2_box_v17.sh agibot_control_functions/_dryrun_box.sh"
Invoke-Robot "ls -lh agibot_control_functions/policies/$PolicyName agibot_control_functions/run_x2_box_v17.sh"

Write-Host ""
Write-Host "On the robot:"
Write-Host "  ssh $($ssh -join ' ')"
Write-Host "  cd ~/agibot_control_functions && ./run_x2_box_v17.sh"
Write-Host "  # first --engage: SUSPENDED, NO BOX"
Write-Host "  ./run_x2_box_v17.sh --engage"
