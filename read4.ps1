$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  Get-ChildItem C:\nova\build\sycl\*.log | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize | Out-String
  $l = "C:\nova\build\sycl\build.log"
  if (Test-Path $l) { "errors=" + ((Select-String -Path $l -Pattern "error:" | Measure-Object).Count); (Get-Content $l -Tail 12 | Out-String) }
} -SessionOption $so | Out-String -Width 200