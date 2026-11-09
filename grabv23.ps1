$d = "\\192.168.137.8\C$\nova\build\sycl"
if (Test-Path "$d\v23_build.log") { Get-Content "$d\v23_build.log" | Select-String "BUILD|DONE" | Out-String }
if (Test-Path "$d\v23_bench.txt") {
  Get-Content "$d\v23_bench.txt" | Select-String -Pattern "GEMV.*GB/s|decode \(tok|attn_decode|attn_prefill|rms_norm|rope|gemv.*rel_l2" | Select-Object -Last 10 | Out-String
}