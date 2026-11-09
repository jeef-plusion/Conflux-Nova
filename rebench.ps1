$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  "exists=" + (Test-Path C:\nova\build\q4k_15b.gguf)
  if (Test-Path C:\nova\build\q4k_15b.gguf) { "size=" + [math]::Round((Get-Item C:\nova\build\q4k_15b.gguf).Length/1MB,1) + " MB" }
  Remove-Item C:\nova\bench_out.txt -ErrorAction SilentlyContinue
  $p = Start-Process -FilePath "C:\Windows\System32\cmd.exe" -ArgumentList "/c","C:\nova\bench.bat" -PassThru -WindowStyle Hidden
  $p.WaitForExit()
  "exit=" + $p.ExitCode
  (Get-Content C:\nova\bench_out.txt -Tail 40 | Out-String)
} -SessionOption $so | Out-String -Width 200