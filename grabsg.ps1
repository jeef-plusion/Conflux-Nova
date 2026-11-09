$d = "\\192.168.137.8\C$\nova\build\sycl"
Get-ChildItem $d -Filter "sg*_bench.txt" -ErrorAction SilentlyContinue | ForEach-Object {
  "=== " + $_.Name + " ==="
  Get-Content $_.FullName | Select-String -Pattern "subgroup|GEMV.*GB/s|decode|dp4a /|rel_l2" | Select-Object -Last 8 | Out-String
}
if (Test-Path "$d\sg_result.txt") { Get-Content "$d\sg_result.txt" }