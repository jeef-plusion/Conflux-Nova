$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 180000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $f = "C:\nova\build\q4k_15b.gguf"
  try { $fs2 = [IO.File]::OpenRead($f); "open ok, len=" + $fs2.Length; $b = New-Object byte[] 4; $fs2.Read($b,0,4) | Out-Null; "magic=" + [Text.Encoding]::ASCII.GetString($b); $fs2.Close() } catch { "open FAIL: " + $_.Exception.Message }
  "attrs=" + (Get-Item $f).Attributes
  "--- run with absolute path ---"
  $o = cmd /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1 && C:\nova\build\sycl\sycl_bench.exe --gguf C:\nova\build\q4k_15b.gguf 2>&1'
  ($o | Select-Object -Last 12 | Out-String)
} -SessionOption $so | Out-String -Width 200