net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "v23.bat" "\\192.168.137.8\C$\nova\v23.bat" -Force
Copy-Item "code\device\sycl\*" "\\192.168.137.8\C$\nova\code\device\sycl\" -Force
"pushed"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 1800000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\v23.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit(); "exit=" + $p.ExitCode
} -SessionOption $so | Out-String