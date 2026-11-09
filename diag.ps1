$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 180000
$sb = {
  "--- run_sycl.bat 原文 ---"
  (Get-Content C:\nova\run_sycl.bat | ForEach-Object { "|" + $_ }) -join "`n"
  "--- vcvars 测试 ---"
  $vc = cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && where cl'
  ($vc | Out-String)
  "--- 直接跑 build_sycl（不经 run_sycl）---"
  $o = cmd /c 'cd /d C:\novauild_sycl.bat 1.5b 2>&1'
  ($o | Select-Object -First 8 | Out-String)
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so | Out-String -Width 200