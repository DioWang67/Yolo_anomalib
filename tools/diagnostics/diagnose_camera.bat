@echo off
setlocal

pushd "%~dp0\..\.."
if exist "yolo11_inference.exe" (
    rem Running from packaged root after build.
) else (
    popd
    pushd "%~dp0"
)

echo === Runtime Check ===
yolo11_inference.exe --check-hikrobot-runtime > runtime_check.log 2>&1

echo === Camera Grab Check ===
yolo11_inference.exe --check-camera-grab > camera_grab_check.log 2>&1

echo === Network Info ===
ipconfig /all > ipconfig_all.log 2>&1
route print > route_print.log 2>&1
netsh advfirewall show allprofiles > firewall_profiles.log 2>&1
netsh advfirewall firewall show rule name=all > firewall_rules.log 2>&1

echo === Adapter Info ===
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetAdapter | Format-List *" > adapter_list.log 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetAdapterBinding -Name * | Format-Table -AutoSize" > adapter_binding.log 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetAdapterAdvancedProperty -Name * | Format-Table -AutoSize" > adapter_advanced.log 2>&1

echo Done.
popd
pause
