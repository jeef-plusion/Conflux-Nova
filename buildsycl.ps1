$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  $cmd = 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul && cd /d C:\nova && build_sycl.bat 1.5b'
  cmd /c $cmd > C:\nova\sycl_build.log 2>&1
  "exit=$LASTEXITCODE"
  (Get-Content C:\nova\sycl_build.log -Tail 45 | Out-String)
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop | Out-String -Width 220