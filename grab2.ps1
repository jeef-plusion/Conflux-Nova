$p = "\\192.168.137.8\C$\nova\build\sycl\build_dp4a.log"
if (Test-Path $p) { Get-Content $p -Tail 6 | Out-String } else { "no build log" }
$b = "\\192.168.137.8\C$\nova\bench_dp4a.txt"
if (Test-Path $b) { "--- bench ---"; Get-Content $b | Select-String -Pattern "dp4a|FAIL|PASS|tok/s|GB/s" | Select-Object -Last 14 | Out-String } else { "no bench yet" }