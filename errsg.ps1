$p = "\\192.168.137.8\C$\nova\build\sycl\sg8_build.log"
if (Test-Path $p) { Get-Content $p | Select-String -Pattern "error" | Select-Object -First 6 | Out-String } else { "no log" }