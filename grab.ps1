$p = "\\192.168.137.8\C$\nova\bench_full.txt"
if (Test-Path $p) { Get-Content $p | Select-Object -Last 45 | Out-String } else { "no file" }