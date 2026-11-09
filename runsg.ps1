net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "sg_experiment.bat" "\\192.168.137.8\C$\nova\sg_experiment.bat" -Force
Copy-Item "code\device\sycl\device.h" "\\192.168.137.8\C$\nova\code\device\sycl\device.h" -Force
"pushed"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 1800000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Remove-Item C:\nova\build\sycl\sg_result.txt -ErrorAction SilentlyContinue
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\sg_experiment.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "exit=" + $p.ExitCode
} -SessionOption $so | Out-String