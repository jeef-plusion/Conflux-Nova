$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
$sb = {
  $p = Get-Process icpx,clang,link,cl,cmd -ErrorAction SilentlyContinue | Select-Object Name,CPU,WS
  "--- processes ---"
  $p | Format-Table -AutoSize | Out-String
  "--- build dir ---"
  if (Test-Path C:\nova\build\sycl) { Get-ChildItem C:\nova\build\sycl | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize | Out-String } else { "no build\sycl" }
  "--- now ---"
  (Get-Date).ToString("HH:mm:ss")
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so | Out-String -Width 200