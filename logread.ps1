$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
$sb = {
  "--- 杀掉卡住的 cmd ---"
  Get-Process cmd -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -lt (Get-Date).AddMinutes(-5) } | Stop-Process -Force -ErrorAction SilentlyContinue
  "killed"
  "--- sycl_build.log ---"
  if (Test-Path C:\nova\sycl_build.log) { (Get-Content C:\nova\sycl_build.log -Tail 25 | Out-String) } else { "no log" }
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so | Out-String -Width 200