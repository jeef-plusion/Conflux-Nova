$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $l = "C:\nova\build\sycl\build2.log"
  if (Test-Path $l) {
    "errors=" + ((Select-String -Path $l -Pattern "error:" | Measure-Object).Count)
    (Get-Content $l -Tail 14 | Out-String)
  } else { "no build2.log" }
  if (Test-Path C:\nova\build\sycl\sycl_bench.exe) { "EXE " + (Get-Item C:\nova\build\sycl\sycl_bench.exe).Length } else { "no exe" }
} -SessionOption $so | Out-String -Width 200