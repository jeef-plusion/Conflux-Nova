net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "make_dp4a.bat" "\\192.168.137.8\C$\nova\make_dp4a.bat" -Force
"pushed"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\make_dp4a.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "exit=" + $p.ExitCode
} -SessionOption $so | Out-String