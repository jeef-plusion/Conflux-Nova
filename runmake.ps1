net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "make.bat" "\\192.168.137.8\C$\nova\make.bat" -Force
"copied"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 600000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c C:\nova\make.bat" -WindowStyle Hidden
  "started"
} -SessionOption $so