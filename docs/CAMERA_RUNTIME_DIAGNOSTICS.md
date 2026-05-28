# Camera Runtime Diagnostics

This project ships a packaged Hikrobot runtime and lightweight diagnostics for
separating packaging issues from camera stream, firewall, and NIC issues.

## Packaged Checks

Run these from the unpacked `yolo11_inference` release folder:

```powershell
.\yolo11_inference.exe --check-hikrobot-runtime
.\yolo11_inference.exe --check-camera-grab
```

`--check-hikrobot-runtime` verifies bundled DLL, CTI, and CLProtocol files and
loads `MvCameraControl.dll`.

`--check-camera-grab` performs a single hardware grab path:

1. Enumerate Hikrobot devices.
2. Open the first camera.
3. Set `TriggerMode=Off`.
4. Set `AcquisitionMode=Continuous`.
5. Print width, height, payload size, and `GevSCPSPacketSize`.
6. Start grabbing.
7. Read one `MV_CC_GetImageBuffer` frame.
8. Stop grabbing and close the camera.

## On-Site Log Collection

Run:

```bat
diagnose_camera.bat
```

It writes these files next to the executable:

```text
runtime_check.log
camera_grab_check.log
ipconfig_all.log
route_print.log
firewall_profiles.log
firewall_rules.log
adapter_list.log
adapter_binding.log
adapter_advanced.log
```

## Interpretation

If `runtime_check.log` fails, treat it as a packaging or DLL search path problem.

If `runtime_check.log` passes but `camera_grab_check.log` shows
`MV_E_NODATA`, `NetReceive[0]`, `socket mode start`, or `DriverVersion[0x0]`,
the bundled user-mode runtime is no longer the first suspect. Compare the
working and failing machines for:

- Inbound UDP firewall rules and endpoint security policy.
- NIC binding differences.
- NIC advanced properties such as Energy Efficient Ethernet, Green Ethernet,
  Interrupt Moderation, Receive Buffers, UDP Checksum Offload, and Large Send
  Offload.
- IP address, subnet, route table, and MTU differences.
- Trigger mode and whether another process is holding the camera.

## Runtime Manifest

Each release build writes:

```text
runtime_manifest_20260528.txt
```

Use it as the current baseline for future runtime comparisons. It records:

- `_internal\Runtime\*.dll`
- `_internal\Runtime\*.cti`
- `_internal\Runtime\CLProtocol\**\*.dll`
- Expected `GENICAM_GENTL64_PATH`
- Expected `MVCAM_GENICAM_CLPROTOCOL`
- Expected runtime entry in `PATH`
