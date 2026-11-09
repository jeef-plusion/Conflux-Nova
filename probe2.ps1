$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  "=== 网络 ==="
  foreach ($h in @("github.com","www.intel.com","pypi.org")) {
    $ok = Test-NetConnection $h -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue
    "$h :443 = $ok"
  }
  "=== winget ==="
  (Get-Command winget -ErrorAction SilentlyContinue).Source
  "=== winget ver ==="
  (winget --version 2>&1 | Select-Object -First 1)
  "=== 已装的相关包 ==="
  winget list --id Microsoft.VisualStudio.2022.BuildTools 2>&1 | Select-Object -Last 2
  "=== python ==="
  (Get-Command python -ErrorAction SilentlyContinue).Source
  "=== GPU 驱动/OpenCL ==="
  Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop | Out-String