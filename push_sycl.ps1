$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 300000
$b64 = "QGVjaG8gb2ZmDQpjYWxsICJDOlxQcm9ncmFtIEZpbGVzICh4ODYpXE1pY3Jvc29mdCBWaXN1YWwgU3R1ZGlvXDIwMjJcQnVpbGRUb29sc1xWQ1xBdXhpbGlhcnlcQnVpbGRcdmN2YXJzNjQuYmF0IiA+bnVsIDI+JjENCmNhbGwgIkM6XFByb2dyYW0gRmlsZXMgKHg4NilcSW50ZWxcb25lQVBJXHNldHZhcnMuYmF0IiA+bnVsIDI+JjENCmNkIC9kIEM6XG5vdmENCmJ1aWxkX3N5Y2wuYmF0IDEuNWINCmVjaG8gX19FWElUX189JUVSUk9STEVWRUwl"
$sb = { param($b64)
  $txt = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b64))
  [IO.File]::WriteAllText("C:\nova\run_sycl.bat", $txt)
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c C:\nova\run_sycl.bat > C:\nova\build_out.log 2>&1" -WindowStyle Hidden
  Start-Sleep -Seconds 8
  if (Test-Path C:\nova\build_out.log) { (Get-Content C:\nova\build_out.log -Tail 15 | Out-String) } else { "no log yet" }
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ArgumentList $b64 -SessionOption $so | Out-String -Width 200