net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
New-Item -ItemType Directory -Force -Path "\\192.168.137.8\C$\nova\build" | Out-Null
$src = "build\abl15b_q4k.gguf"
$dst = "\\192.168.137.8\C$\nova\build\q4k_15b.gguf"
$t0 = Get-Date
Copy-Item $src $dst -Force
"copied in " + [int]((Get-Date) - $t0).TotalSeconds + " s"
"size src=" + (Get-Item $src).Length + " dst=" + (Get-Item $dst).Length