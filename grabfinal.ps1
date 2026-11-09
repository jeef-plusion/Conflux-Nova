$d = "\\192.168.137.8\C$\nova\build\sycl"
if (Test-Path "$d\final_build.log") { Get-Content "$d\final_build.log" | Select-String -Pattern "BUILD|DONE|error:" | Select-Object -First 5 | Out-String }
if (Test-Path "$d\final_bench.txt") {
  Get-Content "$d\final_bench.txt" | Select-String -Pattern "subgroup|local mem|GEMV 有效带宽|decode \(tok|attn_decode|attn_prefill|rel_l2|USM" | Select-Object -Last 12 | Out-String
}