$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  "logsize=" + (Get-Item C:\nova\build\sycl\build.log).Length
  "--- tail 40 ---"
  (Get-Content C:\nova\build\sycl\build.log -Tail 40 | Out-String)
  "--- exe ---"
  if (Test-Path C:\nova\build\sycl\sycl_bench.exe) { "EXE " + (Get-Item C:\nova\build\sycl\sycl_bench.exe).Length + " bytes" } else { "no exe" }
} -SessionOption $so | Out-String -Width 200