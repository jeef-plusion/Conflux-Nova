net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "src\kernel\sycl\*" "\\192.168.137.8\C$\nova\src\kernel\sycl\" -Force
Copy-Item "code\device\sycl\*" "\\192.168.137.8\C$\nova\code\device\sycl\" -Force
"synced"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000 -IdleTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Remove-Item C:\nova\build\sycl\build.log -ErrorAction SilentlyContinue
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\make.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "child exit=" + $p.ExitCode
} -SessionOption $so | Out-String -Width 200