$c = New-Object Net.Sockets.TcpClient
try { $ok = $c.ConnectAsync("192.168.137.8",445).Wait(2500) } catch { $ok = $false }
"tcp445=$ok"
$c.Close()
if ($ok) {
  net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-String
  Copy-Item "run_sycl.bat" "\\192.168.137.8\C$\nova\run_sycl.bat" -Force -ErrorAction Stop
  "copied: " + ((Get-Content "\\192.168.137.8\C$\nova\run_sycl.bat" | Select-Object -Last 2) -join " | ")
}