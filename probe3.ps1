$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  "=== winget search DPC++ ==="
  (winget search --id Intel.oneAPI --accept-source-agreements 2>&1 | Out-String)
  "=== winget search cmake ==="
  (winget search Kitware.CMake 2>&1 | Select-Object -First 6 | Out-String)
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop | Out-String