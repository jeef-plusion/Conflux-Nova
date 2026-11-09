net use \\192.168.137.8\C$ /user:Administrator "kczx05" 2>&1 | Out-Null
Copy-Item "tests\sycl_bench.cpp" "\\192.168.137.8\C$\nova\tests\sycl_bench.cpp" -Force
Copy-Item "src\kernel\sycl\*" "\\192.168.137.8\C$\nova\src\kernel\sycl\" -Force
"synced"
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\make.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "child exit=" + $p.ExitCode
} -SessionOption $so | Out-String