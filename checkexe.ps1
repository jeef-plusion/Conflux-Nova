$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  if (Test-Path C:\nova\build\sycl\sycl_bench.exe) {
    "EXE " + [math]::Round((Get-Item C:\nova\build\sycl\sycl_bench.exe).Length/1MB,2) + " MB  " + (Get-Item C:\nova\build\sycl\sycl_bench.exe).LastWriteTime
  } else { "no exe" }
  (Get-Content C:\nova\build\sycl\ver.txt | Select-Object -First 1)
} -SessionOption $so | Out-String -Width 200