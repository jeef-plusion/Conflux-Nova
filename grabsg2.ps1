$d = "\\192.168.137.8\C$\nova\build\sycl"
if (Test-Path "$d\sg_result.txt") { Get-Content "$d\sg_result.txt" }
foreach ($s in 16,32) {
  $f = "$d\sg${s}_bench.txt"
  if (Test-Path $f) {
    "=== SG=$s ==="
    Get-Content $f | Select-String -Pattern "GEMV 有效带宽|decode \(tok|dp4a /|rel_l2|subgroup" | Select-Object -Last 6 | Out-String
  }
}