$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  if (Test-Path C:\nova\build\sycl\build.log) {
    "logsize=" + (Get-Item C:\nova\build\sycl\build.log).Length
    Get-Content C:\nova\build\sycl\build.log -Tail 25 | Out-String
  } else { "no log yet" }
  if (Test-Path C:\nova\build\sycl\sycl_bench.exe) { "EXE OK " + (Get-Item C:\nova\build\sycl\sycl_bench.exe).Length }
} -SessionOption $so | Out-String -Width 200