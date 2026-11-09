$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  $env:PATH = "C:\Tools\cmake\bin;C:\Tools\MinGit\cmd;" + $env:PATH
  (cmake --version | Select-Object -First 1)
  (git --version)
  $icpx = "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\icpx.exe"
  if (Test-Path $icpx) { (& $icpx --version | Select-Object -First 2) } else { "icpx missing" }
  "--- clone ---"
  if (-not (Test-Path C:\nova)) { git clone --depth 1 https://github.com/jeef-plusion/Conflux-Nova.git C:\nova 2>&1 | Select-Object -Last 2 } else { "already cloned" }
  "--- repo ---"
  Get-ChildItem C:\nova | Select-Object -ExpandProperty Name
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop | Out-String -Width 200