$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  "ver.txt: " + (Test-Path C:\nova\build\sycl\ver.txt)
  if (Test-Path C:\nova\build\sycl\ver.txt) { (Get-Content C:\nova\build\sycl\ver.txt | Select-Object -First 2) -join " / " }
  "make.bat: " + (Test-Path C:\nova\make.bat)
  "icpx proc: " + ((Get-Process icpx,clang -ErrorAction SilentlyContinue | Measure-Object).Count)
  "--- 直接同步跑 icpx --version ---"
  $o = cmd /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1 && icpx --version'
  ($o | Select-Object -First 3 | Out-String)
} -SessionOption $so | Out-String -Width 200