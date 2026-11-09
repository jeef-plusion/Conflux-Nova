$p = "\\192.168.137.8\C$\nova\build\q4k_15b.gguf"
if (Test-Path $p) { $s = (Get-Item $p).Length; "remote=" + [math]::Round($s/1MB,1) + " MB" } else { "not yet" }
"local=" + [math]::Round((Get-Item "build\abl15b_q4k.gguf").Length/1MB,1) + " MB"