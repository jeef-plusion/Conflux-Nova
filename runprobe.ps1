net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "tools\dispatch_probe.cpp" "\\192.168.137.8\C$\nova\tools\dispatch_probe.cpp" -Force
Copy-Item "probe_empty.bat" "\\192.168.137.8\C$\nova\probe_empty.bat" -Force
"pushed"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 600000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\probe_empty.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  if (Test-Path C:\nova\dispatch_out.txt) { (Get-Content C:\nova\dispatch_out.txt | Out-String) } else { "no output"; (Get-Content C:\nova\build\sycl\probe_build.log -Tail 5 | Out-String) }
} -SessionOption $so | Out-String -Width 200