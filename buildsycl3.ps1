$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 900000 -IdleTimeout 900000
$sb = {
  $full = 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1 && cd /d C:\nova && build_sycl.bat 1.5b'
  $out = cmd /c $full 2>&1 | Out-String
  "LEN=" + $out.Length
  ($out -split "`n" | Select-Object -Last 45) -join "`n"
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so -ErrorAction Stop | Out-String -Width 220