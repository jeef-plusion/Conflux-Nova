$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  $o = cmd /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1 && C:\nova\build\sycl\sycl_bench.exe --gguf C:\nova\build\q4k_15b.gguf 2>&1'
  [IO.File]::WriteAllLines("C:\nova\bench_full.txt", $o)
  "lines=" + $o.Count
} -SessionOption $so | Out-String