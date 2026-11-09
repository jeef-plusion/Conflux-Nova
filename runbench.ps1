net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "bench.bat" "\\192.168.137.8\C$\nova\bench.bat" -Force
"pushed bench.bat"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Remove-Item C:\nova\bench_out.txt -ErrorAction SilentlyContinue
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\bench.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "child exit=" + $p.ExitCode
  if (Test-Path C:\nova\bench_out.txt) { "logsize=" + (Get-Item C:\nova\bench_out.txt).Length } else { "no bench log" }
} -SessionOption $so | Out-String