$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000 -IdleTimeout 900000 -OpenTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Remove-Item C:\nova\build\sycl\build.log -ErrorAction SilentlyContinue
  Remove-Item C:\nova\build\sycl\ver.txt -ErrorAction SilentlyContinue
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\make.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "child exit=" + $p.ExitCode
  "ver: " + (Test-Path C:\nova\build\sycl\ver.txt)
  "log: " + (Test-Path C:\nova\build\sycl\build.log)
} -SessionOption $so | Out-String -Width 200