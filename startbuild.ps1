$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 180000
$sb = {
  Remove-Item C:\nova\build_out.log -ErrorAction SilentlyContinue
  Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c C:\nova\run_sycl.bat > C:\nova\build_out.log 2>&1" -WindowStyle Hidden
  Start-Sleep -Seconds 10
  if (Test-Path C:\nova\build_out.log) { "size=" + (Get-Item C:\nova\build_out.log).Length; (Get-Content C:\nova\build_out.log -Tail 8 | Out-String) } else { "no log" }
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so | Out-String -Width 200