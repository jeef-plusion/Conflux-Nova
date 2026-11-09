net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "final.bat" "\\192.168.137.8\C$\nova\final.bat" -Force
Copy-Item "code\device\sycl\*" "\\192.168.137.8\C$\nova\code\device\sycl\" -Force
Copy-Item "src\kernel\sycl\*" "\\192.168.137.8\C$\nova\src\kernel\sycl\" -Force
"pushed"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 1800000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\final.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit(); "exit=" + $p.ExitCode
} -SessionOption $so | Out-String